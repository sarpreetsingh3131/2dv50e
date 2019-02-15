package smc.runmodes;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.json.JSONArray;
import org.json.JSONObject;

import mapek.AdaptationOption;
import mapek.Goal;



public class Comparison extends SMCConnector {
	
	public Comparison() {}
	
	@Override
	public void startVerification() {
		boolean training = cycles <= TRAINING_CYCLE;
		switch (taskType) {
			case CLASSIFICATION:
			case REGRESSION:
				singlePlGoal(training);
				break;
			case PLLAMULTICLASS:
			case PLLAMULTIREGR:
				doublePlLaGoals(training);
				break;
			default:
				throw new RuntimeException(
					String.format("Unsupported task type for Comparison: %s", taskType.val));
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


		// Send the overall results to be saved on the server
		send(adjInspection, TaskType.NONE, Mode.COMPARISON);
	}

	private void doublePlLaGoals(boolean training) {
		if (taskType == TaskType.PLLAMULTIREGR) {
			throw new UnsupportedOperationException("not implemented as of yet");
		}
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


		// Send the overall results to be saved on the server
		send(adjInspection, TaskType.NONE, Mode.COMPARISON);
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

}
