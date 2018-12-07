/*
* Predicts adaption options which fullfill goal(s).
* This is called in the end of the analysis function
* It connects the mapek loop to the model checker and the learner
* Watch out, the goals are also hardcoded in the planning step.
*/


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
import org.json.JSONTokener;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.io.OutputStreamWriter;

import mapek.AdaptationOption;
import mapek.Environment;
import mapek.Link;
import mapek.Mote;
import mapek.SNR;
import mapek.Goal;
import mapek.Goals;
import mapek.TrafficProbability;
import util.ConfigLoader;
import java.io.IOException;
import java.io.File;
import smc.SMCChecker;



// TODO: make it able to abort the whole proces when you receive an error from the ML
// It will make debugging much easier.

public class SMCConnector {

	List<AdaptationOption> adaptationOptions;
	Environment environment;
	SMCChecker smcChecker = new SMCChecker();
	Goals goals = Goals.getInstance();
	List<AdaptationOption> verifiedOptions;

	final int TRAINING_CYCLE = ConfigLoader.getInstance().getAmountOfLearningCycles();
	int cycles = 1;
	Mode mode;
	TaskType taskType;

	// for collecting raw data with the activform mode
	JSONObject rawData;
	JSONArray features;
	JSONArray targets;


	public enum Mode {
		TRAINING("training"), 
		TESTING("testing"), 
		ACTIVFORM("activform"), 
		COMPARISON("comparison"),
		// The new mladjustment mode is similar to the comparison mode.
		// The difference between the two is that mladjustment also checks for
		// the adjustments made to the learners after online learning every cycle.
		// This data is then sent to the python server to be saved in an output file.
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


	public enum TaskType {
		CLASSIFICATION("classification"), 
		REGRESSION("regression"), 

		// I will use this for multiclass classification
		PLLAMULTICLASS("plLaClassification"),
		
		// None is used in case no learning task needs to be performed
		// by the learners at the server side (e.g. when just saving data).
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


	public SMCConnector() {
		// Load the configurations specified in the properties file (mode and tasktype)
		ConfigLoader configLoader = ConfigLoader.getInstance();
		mode = configLoader.getRunMode();
		taskType = configLoader.getTaskType();
		rawData = new JSONObject();
		features = new JSONArray();
		targets = new JSONArray();
		rawData.put("features", features);
		rawData.put("targets", targets);

	}


	public void setAdaptationOptions(List<AdaptationOption> adaptationOptions, Environment environment) {
		this.adaptationOptions = adaptationOptions;
		this.environment = environment;
	}


	public void startVerification() {
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
	 * made to the regression/classification predictions after online learning.
	 * @param training Specifies whether the current cycle is still a training cycle.
	 */
	private void machineLearningAdjustmentInspection(boolean training) {
		JSONObject adjInspection = new JSONObject();
		adjInspection.put("adapIndices", new JSONArray());
		adjInspection.put("packetLoss", new JSONArray());
		adjInspection.put("energyConsumption", new JSONArray());
		adjInspection.put("latency", new JSONArray());
		adjInspection.put("regressionBefore", new JSONArray());
		adjInspection.put("classificationBefore", new JSONArray());
		adjInspection.put("regressionAfter", new JSONArray());
		adjInspection.put("classificationAfter", new JSONArray());

		if (cycles == 1) {
			// At the first cycle, no regression or classification output can be retrieved yet
			// -> use -1 as dummy prediction values
			IntStream.range(0, adaptationOptions.size()).forEach(i -> {
				adjInspection.getJSONArray("regressionBefore").put(-1);
				adjInspection.getJSONArray("classificationBefore").put(-1);
			});
		} else {
			// If not at the first cycle, retrieve the results predicted before online learning
			addPredictionsToJSONArrays(adjInspection.getJSONArray("classificationBefore"),
				adjInspection.getJSONArray("regressionBefore"));
		}


		// Check all the adaptation options with activFORMS
		for (AdaptationOption adaptationOption : adaptationOptions) {
			smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
				adaptationOption.verificationResults);
			adjInspection.getJSONArray("packetLoss").put(adaptationOption.verificationResults.packetLoss);
			adjInspection.getJSONArray("energyConsumption").put(adaptationOption.verificationResults.energyConsumption);
			adjInspection.getJSONArray("latency").put(adaptationOption.verificationResults.latency);
			adjInspection.getJSONArray("adapIndices").put(adaptationOption.overallIndex);
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

			// Parse the classification and regression results from the JSON responses.
			final List<Integer> classificationResults = adjInspection.getJSONArray("classificationBefore")
				.toList().stream()
				.map(o -> Integer.parseInt(o.toString()))
				.collect(Collectors.toList());
			final List<Float> regressionResults = adjInspection.getJSONArray("regressionBefore")
				.toList().stream()
				.map(o -> Float.parseFloat(o.toString()))
				.collect(Collectors.toList());

			Goal pl = goals.getPacketLossGoal();
			// Determine which adaptation options have to be sent back for the specific learners
			// TODO: adopt for latency
			for (int i = 0; i < adaptationOptions.size(); i++) {
				if (classificationResults.get(i).equals(1)) {
					classificationTrainOptions.add(adaptationOptions.get(i));
				}
				if (pl.evaluate(regressionResults.get(i))) {
					regressionTrainOptions.add(adaptationOptions.get(i));
				}
			}

			// In case the adaptation space of a prediction is 0, send all adaptations back for online learning
			if (classificationResults.stream().noneMatch(o -> o == 1)) {
				classificationTrainOptions = adaptationOptions;
			}
			if (regressionResults.stream().noneMatch(o -> pl.evaluate(o))) {
				regressionTrainOptions = adaptationOptions;
			}

			// Send the adaptation options specific to the learners back for online learning
			send(classificationTrainOptions, TaskType.CLASSIFICATION, Mode.TRAINING);
			send(regressionTrainOptions, TaskType.REGRESSION, Mode.TRAINING);

			// Test the predictions of the learners again after online learning to track their adjustments
			addPredictionsToJSONArrays(adjInspection.getJSONArray("classificationAfter"),
				adjInspection.getJSONArray("regressionAfter"));
		}


		// NOTE: experimental, used for feature selection
		// FIXME: remove this after feature selection
		storeAllFeaturesAndTargets();

		// Send the overall results to be saved on the server
		send(adjInspection, TaskType.NONE, Mode.MLADJUSTMENT);
	}


	/**
	 * FIXME remove this later on.
	 */
	private void storeAllFeaturesAndTargets() {
		// Store the features and the targets in their respective files
		File feature_selection = new File(
			Paths.get(System.getProperty("user.dir"), "activforms", "log", "dataset_with_all_features.json").toString());

		if (feature_selection.exists() && cycles == 1) {
			// At the first cycle, remove the file if it already exists
			feature_selection.delete();
			try {
				feature_selection.createNewFile();
				JSONObject root = new JSONObject();
				root.put("features", new JSONArray());
				root.put("target_classification_packetloss", new JSONArray());
				root.put("target_regression_packetloss", new JSONArray());
				root.put("target_classification_latency", new JSONArray());
				root.put("target_regression_latency", new JSONArray());
				FileWriter writer = new FileWriter(feature_selection);
				writer.write(root.toString(2));
				writer.close();
			} catch (IOException e) {
				throw new RuntimeException(
					String.format("Could not create the output file at %s", feature_selection.toPath().toString()));
			}
		}


		try {
			JSONTokener tokener = new JSONTokener(feature_selection.toURI().toURL().openStream());
			JSONObject root = new JSONObject(tokener);

			// Get all the features for all the adaptation options, as well as their targets
			for (AdaptationOption option : adaptationOptions) {
				JSONArray newFeatures = new JSONArray();

				// 17 links (SNR)
				for (SNR snr : environment.linksSNR) {
					newFeatures.put((int) snr.SNR);
				}
				
				// 17 links (Power)
				option.system.motes.values().stream()
					.map(mote -> mote.getLinks())
					.flatMap(links -> links.stream())
					.forEach(link -> newFeatures.put((int) link.getPower()));
				
				// 17 links (Distribution)
				for (Mote mote : option.system.motes.values()) {
					for (Link link : mote.getLinks()) {
						newFeatures.put((int) link.getDistribution());
					}
				}
				
				// 14 motes (Traffic load)
				for (TrafficProbability traffic : environment.motesLoad) {
					newFeatures.put((int) traffic.load);
				}
				
				// => Total of 65 features
				
				root.getJSONArray("features").put(newFeatures);
				root.getJSONArray("target_classification_packetloss").put(
					goals.getPacketLossGoal().evaluate(option.verificationResults.packetLoss) ? 1 : 0);
				root.getJSONArray("target_regression_packetloss").put((int) option.verificationResults.packetLoss);
				
				root.getJSONArray("target_classification_latency").put(
					goals.getLatencyGoal().evaluate(option.verificationResults.latency) ? 1 : 0);
				root.getJSONArray("target_regression_latency").put((int) option.verificationResults.latency);
			}
			FileWriter writer = new FileWriter(feature_selection);
			writer.write(root.toString(2));
			writer.close();
			
		} catch (IOException e) {
			throw new RuntimeException(
				String.format("Could not write to the output file at %s", feature_selection.toPath().toString()));
		}
	}

	void comparison() {

		// Send all the adaptation options to the learners for testing
		JSONObject classificationResponse = send(adaptationOptions, TaskType.CLASSIFICATION, Mode.TESTING);
		JSONObject regressionResponse = send(adaptationOptions, TaskType.REGRESSION, Mode.TESTING);
		
		// Get the size of the predicted adaptation space (for classification and regression)
		int classificationAdaptationSpace = Integer.parseInt(classificationResponse.get("adaptation_space").toString());
		int regressionAdaptationSpace = Integer.parseInt(regressionResponse.get("adaptation_space").toString());
		
		System.out.print(";" + classificationAdaptationSpace + ";" + regressionAdaptationSpace);


		ArrayList<Integer> classificationPredictions = new ArrayList<>();
		ArrayList<Float> regressionPredictions = new ArrayList<>();
		
		// The options which should be used for online learning
		List<AdaptationOption> classificationTrainingOptions = new LinkedList<>();
		List<AdaptationOption> regressionTrainingOptions = new LinkedList<>();

		// Parse the responses for classification and regression
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
			
			// Verify the results using the quality models
			smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
			adaptationOption.verificationResults);
			
			// TODO: adjust to multiple goals
			// If classification predicts that the goal is met (or if no options are predicted to meet the goal)
			if (classificationPredictions.get(index) == 1 || classificationAdaptationSpace == 0) {
				classificationTrainingOptions.add(adaptationOption);
			}

			// Same story as above for regression
			if (goals.getPacketLossGoal().evaluate(regressionPredictions.get(index)) || regressionAdaptationSpace == 0) {
				regressionTrainingOptions.add(adaptationOption);
			}

			// The formally verified options which meet the goal
			if (goals.getPacketLossGoal().evaluate(adaptationOption.verificationResults.packetLoss)) {
				activformAdapationSpace++;
			}

			comparison.getJSONArray("packetLoss").put(adaptationOption.verificationResults.packetLoss);
			comparison.getJSONArray("energyConsumption").put(adaptationOption.verificationResults.energyConsumption);
			
			index++;
		}


		System.out.print(";" + activformAdapationSpace);
		
		// Send back the formally verified options which were predicted to meet the goal by the classifier for online learning
		send(classificationTrainingOptions, TaskType.CLASSIFICATION, Mode.TRAINING);
		// Same story as above for regression
		send(regressionTrainingOptions, TaskType.REGRESSION, Mode.TRAINING);

		// Save the outcome (at server side)
		send(comparison, TaskType.REGRESSION, Mode.COMPARISON);
	}


	void training(TaskType taskType) {
		// Formally verify all the adaptation options, and send them to the learners for training
		for (AdaptationOption adaptationOption : adaptationOptions) {
			smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
					adaptationOption.verificationResults);
		}
		send(adaptationOptions, taskType, Mode.TRAINING);
	}


	void testing(TaskType taskType) {
		// Send the adaptation options to the learner with mode testing, returns the predictions of the learner
		JSONObject response = send(adaptationOptions, taskType, Mode.TESTING);
		
		// Retrieve the amount of options that were predicted to meet the goal by the learner
		int adaptationSpace = Integer.parseInt(response.get("adaptation_space").toString());
		System.out.print(";" + adaptationSpace);

		ArrayList<Float> predictions = new ArrayList<>();
		List<AdaptationOption> qosEstimates = new LinkedList<>();

		// The different prediction classes in case of 2 goals (latency & packet loss)
		int[] mclass = {0, 0, 0, 0};

		JSONArray arr = response.getJSONArray("predictions");
		for (int i = 0; i < arr.length(); i++) {
			if(taskType == TaskType.PLLAMULTICLASS) {
				mclass[Integer.parseInt(arr.get(i).toString())]++;
			}
			predictions.add(Float.parseFloat(arr.get(i).toString()));
		}

		int nbCorrect = 0;
		// Here I set nbCorrect to the highest ammount of 
		// goals predicted correct for every option in the adaption space
		if (taskType == TaskType.PLLAMULTICLASS) {
			if (mclass[3] > 0) {
				// There is at least one option which satisfies both goals
				nbCorrect = 2;
			} else if (mclass[2] + mclass[1] > 0) {
				// There is at least one option which satisfies one of the goals
				nbCorrect = 1;
			}
		}

		
		int i = 0;
		Goal pl = goals.getPacketLossGoal();

		for (AdaptationOption adaptationOption : adaptationOptions) {
			if (adaptationSpace != 0) {
				boolean isPredictedCorrect = false;
				if (taskType == TaskType.CLASSIFICATION) {
					if (predictions.get(i) == 1.0) isPredictedCorrect = true;
				}
				else if(taskType == TaskType.REGRESSION) {
					if (pl.evaluate(predictions.get(i))) isPredictedCorrect = true;
				}
				else if(taskType == TaskType.PLLAMULTICLASS) {
					double pred = predictions.get(i);

					if (nbCorrect == 2) {
						if (pred == 3.0) isPredictedCorrect = true;
					} else if ( nbCorrect == 1) {
						if (pred == 2.0 || pred == 1.0) isPredictedCorrect = true;
					}

				}

				if(isPredictedCorrect) {
					smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
							adaptationOption.verificationResults);

					// Add this option to the list of options that should be sent back for online learning
					qosEstimates.add(adaptationOption);
				} else {
					// The packet loss is manually set to 100 here to make sure this option is never considered.
					adaptationOption.verificationResults.packetLoss = 100.0;
				}
			} else {
				// In case no options were predicted to meet the goals, verify all of them
				smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
						adaptationOption.verificationResults);
				qosEstimates.add(adaptationOption);
			}
			i++;
		}

		// Perform online learning on the samples that were predicted to meet the user goal
		// Note: if no samples were predicted to meet the goal, all the options are sent back for online learning
		send(qosEstimates, taskType, Mode.TRAINING);
	}


	void activform() {
		System.out.print(";" + adaptationOptions.size());
		String datPath = Paths.get(System.getProperty("user.dir"), "activforms", "log", "rawData.txt").toString();
		File dat = new File(datPath);

		try {
			if (cycles == 1 && dat.isFile()) {
				dat.delete();
				dat.createNewFile();
			}
			FileWriter writer = new FileWriter(dat, true);
			writer.write(environment.toString());
			
			JSONArray dummyFeatures;
			JSONObject qos;

			for (AdaptationOption adaptationOption : adaptationOptions) {
				smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
						adaptationOption.verificationResults);
	
				writer.write(adaptationOption.toString());
	
				// Get the features
				dummyFeatures = new JSONArray();

				// The order should be fine since the hashmap is a LinkedHashMap (follows insertion order)
				for (Mote mote : adaptationOption.system.motes.values()) {
					
					for (TrafficProbability t : environment.motesLoad) {
						if(t.moteId == mote.getMoteId()) {
							dummyFeatures.put(t.load);
						}
					}
	
					for (Link link : mote.getLinks()) {
						dummyFeatures.put(link.getPower());
						dummyFeatures.put(link.getDistribution());
						dummyFeatures.put(environment.getSNR(link));
					}
				}
				features.put(dummyFeatures);
	
	
				// get qos
				// This has to be processed for every mode, thats why I add objects.
				qos = new JSONObject();
				qos.put("packetLoss", adaptationOption.verificationResults.packetLoss);
				qos.put("latency", adaptationOption.verificationResults.latency);
				qos.put("energyConsumption", adaptationOption.verificationResults.energyConsumption);
				targets.put(qos);
			}
			writer.close();
			
		} catch(Exception e) {
			throw new RuntimeException("Failed to write to the raw data file (activform).");
		}
		
		// write at the end of all the cycles
		if (this.cycles == ConfigLoader.getInstance().getAmountOfCycles()) {
			Path p = Paths.get(Paths.get(System.getProperty("user.dir")).toString(), "activforms", "log");
			
			// find none existing file
			int i = 1;
			File f = null;
			int cyc = ConfigLoader.getInstance().getAmountOfCycles();
			int dist = ConfigLoader.getInstance().getDistributionGap();

			while (true) {
				f = new File(Paths.get(p.toString(), 
					String.format("%iCycles%iDist_run%i.json", cyc, dist, i)).toString());

				if(!f.exists()) {
					break;
				} else {
					i++;
				}
			}

			try {
				FileWriter jsonWriter = new FileWriter(f);
				jsonWriter.write(rawData.toString());
				jsonWriter.flush();
				jsonWriter.close();
			} catch (Exception e) {
				System.out.println("Problem writing to file.\n");
			}
		}
	}


	/**
	 * Prepare the adaptation options (their features and targets) for the machine learner.
	 * @param adaptationOptions the options which should be prepared.
	 * @param taskType the task type (which is necessary to decide the target).
	 * @return a JSONObject which contains the data that should be sent to the learner.
	 */
	JSONObject parse(List<AdaptationOption> adaptationOptions, TaskType taskType) {
		
		JSONObject dataset = new JSONObject();
		JSONArray features = new JSONArray();
		JSONArray target = new JSONArray();

		dataset.put("features", features);
		dataset.put("target", target);

		for (AdaptationOption adaptationOption : adaptationOptions) {
			JSONArray item = new JSONArray();
			Goal pl = goals.getPacketLossGoal();

			// Decide the target for the adaptation option (dependent on the task type)
			if (taskType == TaskType.CLASSIFICATION) {
				target.put(pl.evaluate(adaptationOption.verificationResults.packetLoss) ? 1 : 0);
			} else if (taskType == TaskType.REGRESSION) {
				target.put((int) adaptationOption.verificationResults.packetLoss);
			} else if(taskType == TaskType.PLLAMULTICLASS) {
				// makes corresponding classes for multiclass verification
				// There are 8 possible combinations for the goals
				// The succes of a goal represent one bit in 
				// 3 bits (a power of 2).
				// So all possible combination can be represented in 3 bits.
				// So a  number from 0-7 = class.
				int APClass = 0;

				Goal la = goals.getLatencyGoal();

				if (pl.evaluate(adaptationOption.verificationResults.packetLoss)) APClass += 1;
				if (la.evaluate(adaptationOption.verificationResults.latency)) APClass += 2;

				target.put(APClass);
			}
			
			// Add the SNR values of all the links in the environment
			for (SNR snr : environment.linksSNR) {
				item.put((int) snr.SNR);
			}

			// Add the power settings for all the links
			// TODO do feature selection again because of this part
			adaptationOption.system.motes.values().stream()
				.map(mote -> mote.getLinks())
				.flatMap(links -> links.stream())
				.forEach(link -> item.put((int) link.getPower()));

			// Add the distribution values for the links from motes 7, 10 and 12
			for (Mote mote : adaptationOption.system.motes.values()) {
				for (Link link : mote.getLinks()) {
					if (link.getSource() == 7 || link.getSource() == 10 || link.getSource() == 12) {
						item.put((int) link.getDistribution());
					}
				}
			}

			// Add the load for motes 10 and 12
			for (TrafficProbability traffic : environment.motesLoad) {
				if (traffic.moteId == 10 || traffic.moteId == 12) {
					item.put((int) traffic.load);
				}
			}

			features.put(item);
		}

		// Returns the dataset with the features and targets
		return dataset;
	}


	/**
	 * This method parses the adaptation options to a JSONObject and sends them to the webserver which runs the machine learner.
	 * See {@link #send(JSONObject, String, String) send(JSONObject, String, String)} for more information.
	 */
	private JSONObject send(List<AdaptationOption> adaptationOptions, TaskType taskType, Mode mode) {
		return send(parse(adaptationOptions, taskType), taskType, mode);
	}


	/**
	 * See {@link #send(JSONObject, String, String) send(JSONObject, String, String)} for more information.
	 */
	private JSONObject send(JSONObject dataset, TaskType taskType, Mode mode) {
		return send(dataset, taskType.val, mode.val);
	}


	/**
	 * Sends the provided dataset to the webserver running the machine learner.
	 * @param dataset The data set which should be sent over to the server.
	 * @param taskType The task that needs to be performed (e.g. testing).
	 * @param mode The mode associated with the task (e.g. classification).
	 * @return The response from the server.
	 */
	JSONObject send(JSONObject dataset, String taskType, String mode) {
		try {
			HttpClient client = HttpClientBuilder.create().build();
			
			// Create a post message for the webserver (which runs the machine learner)
			// The server responds with predictions (in case the task type is testing), or OK/NOK for other tasks
			HttpPost http = new HttpPost("http://localhost:8000/?task-type=" + taskType + "&mode=" + mode + "&cycle=" + cycles);
			
			// The entity of a http post is the payload
			// Here you give as payload the json dataset to send in the form of a string
			http.setEntity(new StringEntity(dataset.toString()));
			http.setHeader("Content-Type", "application/json");
			
			// Keep track of how long it takes to perform the given task at the end of the webserver
			long start = System.currentTimeMillis();

			JSONObject response = new JSONObject(client.execute(http, new BasicResponseHandler()));
			
			// Print the time it took for the webserver to send an answer back
			System.out.print(";" + (System.currentTimeMillis() - start));

			return response;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}
}


