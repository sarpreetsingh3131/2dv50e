package smc;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.BasicResponseHandler;
import org.apache.http.impl.client.HttpClientBuilder;
import org.json.JSONArray;
import org.json.JSONObject;

import mapek.AdaptationOption;
import mapek.Environment;
import mapek.Link;
import mapek.Mote;
import mapek.SNR;
import mapek.TrafficProbability;

public class SMCConnector {

	List<AdaptationOption> adaptationOptions;
	Environment environment;
	SMCChecker smcChecker = new SMCChecker();
	List<AdaptationOption> verifiedOptions;
	final int TRAINING_CYCLE = 5;
	int cycles = 1;

	enum Mode {
		TRAINING("training"), 
		TESTING("testing"), 
		ACTIVFORM(""), 
		COMPARISON("comparison"),
		MLADJUSTMENT("mladjustment");

		String val;

		Mode(String val) {
			this.val = val;
		}

		public static Mode getMode(String value) {
			for (Mode mode : Mode.values()) {
				if (mode.val.equals(value)) {
					return mode;
				}
			}
			throw new RuntimeException(String.format("Run mode %s is not supported.", value));
		}
	}

	enum TaskType {
		CLASSIFICATION("classification"), 
		REGRESSION("regression"), 
		NONE("none");
		String val;

		TaskType(String val) {
			this.val = val;
		}

		public static TaskType getTaskType(String value) {
			for (TaskType taskType : TaskType.values()) {
				if (taskType.val.equals(value)) {
					return taskType;
				}
			}
			throw new RuntimeException(String.format("Task type %s is not supported.", value));
		}
	}

	public void setAdaptationOptions(List<AdaptationOption> adaptationOptions, Environment environment) {
		this.adaptationOptions = adaptationOptions;
		this.environment = environment;
		// System.out.println("Environment:" + environment);

	}

	public void startVerification() {
		// System.out.println("Verification started!");
		Mode mode = Mode.ACTIVFORM;
		TaskType taskType = TaskType.REGRESSION;

		switch (mode) {
		case ACTIVFORM:
			activform();
			break;
		case COMPARISON:
			if (cycles <= TRAINING_CYCLE) {
				int space = 0;
				for (AdaptationOption adaptationOption : adaptationOptions) {
					smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
							adaptationOption.verificationResults);
					if (adaptationOption.verificationResults.packetLoss < 10.0) {
						space++;
					}
				}
				System.out.print(";" + space);
				send(adaptationOptions, TaskType.CLASSIFICATION, Mode.TRAINING);
				send(adaptationOptions, TaskType.REGRESSION, Mode.TRAINING);

			} else
				comparison();
			break;
		case MLADJUSTMENT:
			machineLearningAdjustmentInspection(cycles <= TRAINING_CYCLE);
			break;
		default:
			if (cycles <= TRAINING_CYCLE)
				training(taskType);
			else
				testing(taskType);
		}
		cycles++;
	}

/**
	 * Helper function which adds the predictions of both learning models to their respective JSON arrays.
	 * The predictions are made over the whole adaptation space.
	 * @param classArray The JSON array which will hold the classification predictions.
	 * @param regrArray The JSON array which will hold the regression predictions.
	 */
	private void addPredictionsToJSONArrays(JSONArray classArray, JSONArray regrArray) {
		JSONObject classificationResponse = send(adaptationOptions, TaskType.CLASSIFICATION, Mode.TESTING);
		JSONObject regressionResponse = send(adaptationOptions, TaskType.REGRESSION, Mode.TESTING);

		for (Object item : classificationResponse.getJSONArray("predictions")) {
			classArray.put(Integer.parseInt(item.toString()));
		}
		for (Object item : regressionResponse.getJSONArray("predictions")) {
			regrArray.put(Float.parseFloat(item.toString()));
		}
	}

	/**
	 * FIXME remove later on, here for data analysis
	 *
	 * Similar execution as comparison, but also tracks the adjustments
	 * made to the regression/classifications predictions after online learning.
	 * @param training Specifies whether the current cycle is still a training cycle.
	 */
	private void machineLearningAdjustmentInspection(boolean training) {
		JSONObject adjInspection = new JSONObject();
		adjInspection.put("packetLoss", new JSONArray());
		adjInspection.put("energyConsumption", new JSONArray());
		adjInspection.put("regressionBefore", new JSONArray());
		adjInspection.put("classificationBefore", new JSONArray());
		adjInspection.put("regressionAfter", new JSONArray());
		adjInspection.put("classificationAfter", new JSONArray());

		if (cycles == 1) {
			// At the first cycle, no regression or classification output can be retrieved yet
			IntStream.range(0, adaptationOptions.size()).forEach(i -> {
				adjInspection.getJSONArray("regressionBefore").put(-1);
				adjInspection.getJSONArray("classificationBefore").put(-1);
			});
		} else {
			// If not at the first cycle, retrieve the results predicted before learning
			addPredictionsToJSONArrays(adjInspection.getJSONArray("classificationBefore"),
				adjInspection.getJSONArray("regressionBefore"));
		}


		// Check all the adaptation options
		for (AdaptationOption adaptationOption : adaptationOptions) {
			smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
				adaptationOption.verificationResults);
			adjInspection.getJSONArray("packetLoss").put(adaptationOption.verificationResults.packetLoss);
			adjInspection.getJSONArray("energyConsumption").put(adaptationOption.verificationResults.energyConsumption);
		}

		if (training) {
			// If we are training, send the entire adaptation space to the learners and check what they have learned
			send(adaptationOptions, TaskType.CLASSIFICATION, Mode.TRAINING);
			send(adaptationOptions, TaskType.REGRESSION, Mode.TRAINING);

			addPredictionsToJSONArrays(adjInspection.getJSONArray("classificationAfter"),
				adjInspection.getJSONArray("regressionAfter"));
		} else {
			// If we are testing, send the adjustments to the learning models and check their predictions again
			List<AdaptationOption> classificationTrainOptions = new ArrayList<>();
			List<AdaptationOption> regressionTrainOptions = new ArrayList<>();
			final List<Integer> classificationResults = adjInspection.getJSONArray("classificationBefore")
				.toList().stream()
				.map(o -> Integer.parseInt(o.toString()))
				.collect(Collectors.toList());
			final List<Float> regressionResults = adjInspection.getJSONArray("regressionBefore")
				.toList().stream()
				.map(o -> Float.parseFloat(o.toString()))
				.collect(Collectors.toList());

			// Determine which adaptation options have to be sent back for the learning cycle
			for (int i = 0; i < adaptationOptions.size(); i++) {
				if (classificationResults.get(i).equals(1)) {
					classificationTrainOptions.add(adaptationOptions.get(i));
				}
				if (regressionResults.get(i) < 10) {
					regressionTrainOptions.add(adaptationOptions.get(i));
				}
			}

			// In case the adaptation space of a prediction is 0, send all adaptations back
			if (classificationResults.stream().noneMatch(o -> o == 1)) {
				classificationTrainOptions = adaptationOptions;
			}
			if (regressionResults.stream().noneMatch(o -> o < 10)) {
				regressionTrainOptions = adaptationOptions;
			}

			send(classificationTrainOptions, TaskType.CLASSIFICATION, Mode.TRAINING);
			send(regressionTrainOptions, TaskType.REGRESSION, Mode.TRAINING);
			addPredictionsToJSONArrays(adjInspection.getJSONArray("classificationAfter"),
				adjInspection.getJSONArray("regressionAfter"));
		}

		send(adjInspection, TaskType.NONE, Mode.MLADJUSTMENT);
	}



	void comparison() {
		JSONObject classificationResponse = send(adaptationOptions, TaskType.CLASSIFICATION, Mode.TESTING);
		JSONObject regressionResponse = send(adaptationOptions, TaskType.REGRESSION, Mode.TESTING);
		int classificationAdaptationSpace = Integer.parseInt(classificationResponse.get("adaptation_space").toString());
		int regressionAdaptationSpace = Integer.parseInt(regressionResponse.get("adaptation_space").toString());
		System.out.print(";" + classificationAdaptationSpace + ";" + regressionAdaptationSpace);

		ArrayList<Integer> classificationPredictions = new ArrayList<>();
		ArrayList<Float> regressionPredictions = new ArrayList<>();
		List<AdaptationOption> classificationTrainingOptions = new LinkedList<>();
		List<AdaptationOption> regressionTrainingOptions = new LinkedList<>();

		JSONArray arr = classificationResponse.getJSONArray("predictions");
		for (int i = 0; i < arr.length(); i++) {
			classificationPredictions.add(Integer.parseInt(arr.get(i).toString()));
		}

		arr = regressionResponse.getJSONArray("predictions");
		for (int i = 0; i < arr.length(); i++) {
			regressionPredictions.add(Float.parseFloat(arr.get(i).toString()));
		}

		int activformAdapationSpace = 0;
		int index = 0;
		JSONObject comparison = new JSONObject();
		comparison.put("packetLoss", new JSONArray());
		comparison.put("energyConsumption", new JSONArray());
		comparison.put("classification", classificationResponse.getJSONArray("predictions"));
		comparison.put("regression", regressionResponse.getJSONArray("predictions"));

		for (AdaptationOption adaptationOption : adaptationOptions) {
			smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
					adaptationOption.verificationResults);
			if (classificationPredictions.get(index) == 1 || classificationAdaptationSpace == 0) {
				classificationTrainingOptions.add(adaptationOption);
			}
			if (regressionPredictions.get(index) < 10.0 || regressionAdaptationSpace == 0) {
				regressionTrainingOptions.add(adaptationOption);
			}
			if (adaptationOption.verificationResults.packetLoss < 10.0) {
				activformAdapationSpace++;
			}
			comparison.getJSONArray("packetLoss").put(adaptationOption.verificationResults.packetLoss);
			comparison.getJSONArray("energyConsumption").put(adaptationOption.verificationResults.energyConsumption);
			index++;
		}
		System.out.print(";" + activformAdapationSpace);
		send(classificationTrainingOptions, TaskType.CLASSIFICATION, Mode.TRAINING);
		send(regressionTrainingOptions, TaskType.REGRESSION, Mode.TRAINING);
		send(comparison, TaskType.REGRESSION, Mode.COMPARISON);
	}

	void training(TaskType taskType) {
		// int space = 0;
		for (AdaptationOption adaptationOption : adaptationOptions) {
			smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
					adaptationOption.verificationResults);
			// if (adaptationOption.verificationResults.packetLoss < 10.0) {
			// space++;
			// }
		}
		// System.out.print(";" + space);
		send(adaptationOptions, taskType, Mode.TRAINING);
	}

	void testing(TaskType taskType) {
		JSONObject response = send(adaptationOptions, taskType, Mode.TESTING);
		int adaptationSpace = Integer.parseInt(response.get("adaptation_space").toString());
		System.out.print(";" + adaptationSpace);

		ArrayList<Float> predictions = new ArrayList<>();
		List<AdaptationOption> qosEstimates = new LinkedList<>();

		JSONArray arr = response.getJSONArray("predictions");
		for (int i = 0; i < arr.length(); i++) {
			predictions.add(Float.parseFloat(arr.get(i).toString()));
		}

		int i = 0;
		for (AdaptationOption adaptationOption : adaptationOptions) {
			if (adaptationSpace != 0) {
				if (taskType == TaskType.CLASSIFICATION ? predictions.get(i) == 1.0 : predictions.get(i) < 10.0) {
					smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
							adaptationOption.verificationResults);
					qosEstimates.add(adaptationOption);
				} else {
					adaptationOption.verificationResults.packetLoss = 100.0;
				}
			} else {
				smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
						adaptationOption.verificationResults);
				qosEstimates.add(adaptationOption);
			}
			i++;
		}
		send(qosEstimates, taskType, Mode.TRAINING);
	}

	void activform() {
		System.out.print(";" + adaptationOptions.size());
		for (AdaptationOption adaptationOption : adaptationOptions) {
			smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
					adaptationOption.verificationResults);
		}
	}

	JSONObject parse(List<AdaptationOption> adaptationOptions, TaskType taskType) {
		JSONObject dataset = new JSONObject();
		JSONArray features = new JSONArray();
		JSONArray target = new JSONArray();

		dataset.put("features", features);
		dataset.put("target", target);

		for (AdaptationOption adaptationOption : adaptationOptions) {
			JSONArray item = new JSONArray();

			if (taskType == TaskType.CLASSIFICATION) {
				target.put(adaptationOption.verificationResults.packetLoss < 10.0 ? 1 : 0);
			} else {
				target.put((int) adaptationOption.verificationResults.packetLoss);
			}

			for (SNR snr : environment.linksSNR) {
				item.put((int) snr.SNR);
			}

			for (Mote mote : adaptationOption.system.motes.values()) {
				for (Link link : mote.getLinks()) {
					if (link.getSource() == 7 || link.getSource() == 10 || link.getSource() == 12) {
						item.put((int) link.getDistribution());
					}
				}
			}

			for (TrafficProbability traffic : environment.motesLoad) {
				if (traffic.moteId == 10 || traffic.moteId == 12) {
					item.put((int) traffic.load);
				}
			}

			features.put(item);
		}
		return dataset;
	}


	private JSONObject send(List<AdaptationOption> adaptationOptions, TaskType taskType, Mode mode) {
		return send(parse(adaptationOptions, taskType), taskType, mode);
	}

	private JSONObject send(JSONObject dataset, TaskType taskType, Mode mode) {
		return send(dataset, taskType.val, mode.val);
	}

	JSONObject send(JSONObject dataset, String taskType, String mode) {
		try {
			HttpClient client = HttpClientBuilder.create().build();
			HttpPost http = new HttpPost("http://localhost:8000/?task-type=" + taskType + "&mode=" + mode + "&cycle=" + cycles);
			http.setEntity(new StringEntity(dataset.toString()));
			http.setHeader("Content-Type", "application/json");
			long start = System.currentTimeMillis();
			JSONObject response = new JSONObject(client.execute(http, new BasicResponseHandler()));
			System.out.print(";" + (System.currentTimeMillis() - start));
			return response;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}
}