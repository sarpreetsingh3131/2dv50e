package smc;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
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
	final int TRAINING_CYCLE = 1;
	int cycles = 1;

	enum Mode {
		TRAINING("training"), TESTING("testing"), ACTIVFORM(""), COMPARISON("comparison");
		String val;

		Mode(String val) {
			this.val = val;
		}
	}

	enum TaskType {
		CLASSIFICATION("classification"), REGRESSION("regression");
		String val;

		TaskType(String val) {
			this.val = val;
		}
	}

	public void setAdaptationOptions(List<AdaptationOption> adaptationOptions, Environment environment) {
		this.adaptationOptions = adaptationOptions;
		this.environment = environment;
		// System.out.println("Environment:" + environment);

	}

	public void startVerification() {
		// System.out.println("Verification started!");
		Mode mode = Mode.TESTING;
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
				send(parse(adaptationOptions, TaskType.CLASSIFICATION), TaskType.CLASSIFICATION, Mode.TRAINING);
				send(parse(adaptationOptions, TaskType.REGRESSION), TaskType.REGRESSION, Mode.TRAINING);

			} else
				comparison();
			break;
		default:
			if (cycles <= TRAINING_CYCLE)
				training(taskType);
			else
				testing(taskType);
		}
		cycles++;
	}

	void comparison() {
		JSONObject classificationResponse = send(parse(adaptationOptions, TaskType.CLASSIFICATION),
				TaskType.CLASSIFICATION, Mode.TESTING);
		JSONObject regressionResponse = send(parse(adaptationOptions, TaskType.REGRESSION), TaskType.REGRESSION,
				Mode.TESTING);
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
		send(parse(classificationTrainingOptions, TaskType.CLASSIFICATION), TaskType.CLASSIFICATION, Mode.TRAINING);
		send(parse(regressionTrainingOptions, TaskType.REGRESSION), TaskType.REGRESSION, Mode.TRAINING);
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
		send(parse(adaptationOptions, taskType), taskType, Mode.TRAINING);
	}

	void testing(TaskType taskType) {
		JSONObject response = send(parse(adaptationOptions, taskType), taskType, Mode.TESTING);
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
		send(parse(qosEstimates, taskType), taskType, Mode.TRAINING);
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

	JSONObject send(JSONObject dataset, TaskType taskType, Mode mode) {
		try {
			HttpClient client = HttpClientBuilder.create().build();
			HttpPost http = new HttpPost("http://localhost:8000/?task-type=" + taskType.val + "&mode=" + mode.val);
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