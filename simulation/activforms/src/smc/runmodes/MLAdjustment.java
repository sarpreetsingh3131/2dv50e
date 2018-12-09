package smc.runmodes;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;

import mapek.AdaptationOption;
import mapek.Environment;
import mapek.Goal;
import mapek.Link;
import mapek.Mote;
import mapek.SNR;
import mapek.TrafficProbability;



public class MLAdjustment extends SMCConnector {
	
	public MLAdjustment() {}
	
	@Override
	public void startVerification() {
		boolean training = cycles <= TRAINING_CYCLE;
		switch (taskType) {
			case CLASSIFICATION:
			case REGRESSION:
				singlePlGoal(training);
				break;
			case PLLAMULTICLASS:
				doublePlLaGoals(training);
				break;
			default:
				throw new RuntimeException(
					String.format("Unsupported task type for MLAdjustment: %s", taskType.val));
		}
	}


	private void singlePlGoal(boolean training) {
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
			predictionLearners1Goal(adaptationOptions, adjInspection.getJSONArray("classificationBefore"),
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

			predictionLearners1Goal(adaptationOptions, adjInspection.getJSONArray("classificationAfter"),
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
			predictionLearners1Goal(adaptationOptions, adjInspection.getJSONArray("classificationAfter"),
				adjInspection.getJSONArray("regressionAfter"));
		}


		// NOTE: experimental, used for feature selection
		// FIXME: remove this after feature selection
		storeAllFeaturesAndTargets(adaptationOptions, environment, cycles);
		
		// Send the overall results to be saved on the server
		send(adjInspection, TaskType.NONE, Mode.MLADJUSTMENT);
	}

	private void doublePlLaGoals(boolean training) {
		// FIXME: dummy values for regression for now (only classification considered atm)
		// TODO: once regression is also used, unify both methods

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
				// JSONArray regrValues = new JSONArray();
				// regrValues.put(-1);
				// regrValues.put(-1);
				adjInspection.getJSONArray("regressionBefore").put(-1);
				adjInspection.getJSONArray("classificationBefore").put(-1);
			});
		} else {
			// If not at the first cycle, retrieve the results predicted before online learning
			predictionLearners2Goals(adaptationOptions, adjInspection.getJSONArray("classificationBefore"),
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
			send(adaptationOptions, TaskType.PLLAMULTICLASS, Mode.TRAINING);
			// send(adaptationOptions, TaskType.REGRESSION, Mode.TRAINING);

			predictionLearners2Goals(adaptationOptions, adjInspection.getJSONArray("classificationAfter"),
				adjInspection.getJSONArray("regressionAfter"));
		} else {
			// If we  are testing, send the adjustments to the learning models and check their predictions again
			List<AdaptationOption> classificationTrainOptions = new ArrayList<>();
			// List<AdaptationOption> regressionTrainOptions = new ArrayList<>();

			// Parse the classification and regression results from the JSON responses.
			final List<Integer> classificationResults = adjInspection.getJSONArray("classificationBefore")
				.toList().stream()
				.map(o -> Integer.parseInt(o.toString()))
				.collect(Collectors.toList());
			// final List<Float> regressionResults = adjInspection.getJSONArray("regressionBefore")
			// 	.toList().stream()
			// 	.map(o -> Float.parseFloat(o.toString()))
			// 	.collect(Collectors.toList());

			// Goal pl = goals.getPacketLossGoal();
			// Goal la = goals.getLatencyGoal();

			// Determine which adaptation options have to be sent back for the specific learners
			// Count the amount of predictions for each classification class
			int predictionsInClass[] = new int[4];
			for (Integer pred : classificationResults) {
				predictionsInClass[pred] += 1;
			}

			int satisfiedGoals = 0;
			if (predictionsInClass[3] > 0) {
				satisfiedGoals = 2;
			} else if (predictionsInClass[1] + predictionsInClass[2] > 0) {
				satisfiedGoals = 1;
			}

			for (int i = 0; i < adaptationOptions.size(); i++) {
				int classResult = classificationResults.get(i);
				if (satisfiedGoals == 2 && classResult == 3) {
					classificationTrainOptions.add(adaptationOptions.get(i));
				} else if (satisfiedGoals == 1 && (classResult == 2 || classResult == 1)) {
					classificationTrainOptions.add(adaptationOptions.get(i));
				} else if (satisfiedGoals == 0) {
					classificationTrainOptions.add(adaptationOptions.get(i));
				}
			}

			// In case the adaptation space of a prediction is 0, send all adaptations back for online learning
			if (satisfiedGoals == 0) {
				classificationTrainOptions = adaptationOptions;
			}
			// if (regressionResults.stream().noneMatch(o -> pl.evaluate(o))) {
			// 	regressionTrainOptions = adaptationOptions;
			// }

			// Send the adaptation options specific to the learners back for online learning
			send(classificationTrainOptions, TaskType.PLLAMULTICLASS, Mode.TRAINING);
			// send(regressionTrainOptions, TaskType.REGRESSION, Mode.TRAINING);

			// Test the predictions of the learners again after online learning to track their adjustments
			predictionLearners2Goals(adaptationOptions, adjInspection.getJSONArray("classificationAfter"),
				adjInspection.getJSONArray("regressionAfter"));
		}


		// NOTE: experimental, used for feature selection
		// FIXME: remove this after feature selection
		storeAllFeaturesAndTargets(adaptationOptions, environment, cycles);

		// Send the overall results to be saved on the server
		send(adjInspection, TaskType.NONE, Mode.MLADJUSTMENT);
	}



	/**
	 * Helper function which adds the predictions of both learning models to their respective JSON arrays.
	 * The predictions are made over the whole adaptation space.
	 * @param classArray The JSON array which will hold the classification predictions.
	 * @param regrArray The JSON array which will hold the regression predictions.
	 */
	private void predictionLearners1Goal(List<AdaptationOption> adaptationOptions, JSONArray classArray, JSONArray regrArray) {
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
	 * Helper function which adds the predictions of both learning models to their respective JSON arrays.
	 * The predictions are made over the whole adaptation space.
	 * @param classArray The JSON array which will hold the classification predictions.
	 * @param regrArray The JSON array which will hold the regression predictions.
	 */
	private void predictionLearners2Goals(List<AdaptationOption> adaptationOptions, JSONArray classArray, JSONArray regrArray) {
		JSONObject classificationResponse = send(adaptationOptions, TaskType.PLLAMULTICLASS, Mode.TESTING);
		// JSONObject regressionResponse = send(adaptationOptions, TaskType.REGRESSION, Mode.TESTING);

		for (Object item : classificationResponse.getJSONArray("predictions")) {
			classArray.put(Integer.parseInt(item.toString()));
		}
		// TODO: dummy values for now
		for (int i = 0; i < classificationResponse.getJSONArray("predictions").length(); i++) {
			regrArray.put(-1);
		}
	}

	private void storeAllFeaturesAndTargets(List<AdaptationOption> adaptationOptions, Environment env, int cycle) {
		// Store the features and the targets in their respective files
		File feature_selection = new File(
			Paths.get(System.getProperty("user.dir"), "activforms", "log", "dataset_with_all_features.json").toString());

		if (feature_selection.exists() && cycle == 1) {
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
				for (SNR snr : env.linksSNR) {
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
				for (TrafficProbability traffic : env.motesLoad) {
					newFeatures.put((int) traffic.load);
				}
				
				// => Total of 65 features
				
				// Features
				root.getJSONArray("features").put(newFeatures);
				
				// Packet loss values
				root.getJSONArray("target_classification_packetloss").put(
					goals.getPacketLossGoal().evaluate(option.verificationResults.packetLoss) ? 1 : 0);
				root.getJSONArray("target_regression_packetloss").put((int) option.verificationResults.packetLoss);
				
				// Latency values
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
}
