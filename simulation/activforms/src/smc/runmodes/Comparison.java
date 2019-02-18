package smc.runmodes;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.json.JSONArray;
import org.json.JSONObject;

import mapek.AdaptationOption;
import mapek.Goal;
import mapek.Goals;
import util.ConfigLoader;



public class Comparison extends SMCConnector {

	private int lastLearningIndex = 0;

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
		adjInspection.put("regressionPLBefore", new JSONArray());
		adjInspection.put("classificationBefore", new JSONArray());
		adjInspection.put("regressionPLAfter", new JSONArray());
		adjInspection.put("classificationAfter", new JSONArray());

		if (cycles == 1) {
			// At the first cycle, no regression or classification output can be retrieved yet
			// -> use -1 as dummy prediction values
			IntStream.range(0, adaptationOptions.size()).forEach(i -> {
				adjInspection.getJSONArray("regressionPLBefore").put(-1);
				adjInspection.getJSONArray("classificationBefore").put(-1);
			});
		} else {
			// If not at the first cycle, retrieve the results predicted before online learning
			predictionLearners1Goal(adaptationOptions, adjInspection.getJSONArray("classificationBefore"),
				adjInspection.getJSONArray("regressionPLBefore"));
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
				adjInspection.getJSONArray("regressionPLAfter"));
		} else {
			// If we are testing, send the adjustments to the learning models and check their predictions again
			List<AdaptationOption> classificationTrainOptions = new ArrayList<>();
			List<AdaptationOption> regressionTrainOptions = new ArrayList<>();

			// Parse the classification and regression results from the JSON responses.
			final List<Integer> classificationResults = adjInspection.getJSONArray("classificationBefore")
				.toList().stream()
				.map(o -> Integer.parseInt(o.toString()))
				.collect(Collectors.toList());
			final List<Float> regressionResults = adjInspection.getJSONArray("regressionPLBefore")
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
				adjInspection.getJSONArray("regressionPLAfter"));
		}


		// Send the overall results to be saved on the server
		send(adjInspection, TaskType.NONE, Mode.COMPARISON);
	}

	private void addPredictionsBeforeLearning(JSONArray cl, JSONArray rePL, JSONArray reLA) {
		if (cycles == 1) {
			// At the first cycle, no regression or classification output can be retrieved yet
			// -> use -1 as dummy prediction values
			IntStream.range(0, adaptationOptions.size()).forEach(i -> {
				cl.put(-1);
				rePL.put(-1);
				reLA.put(-1);
			});
		} else {
			// If not at the first cycle, retrieve the results predicted before online learning
			predictionLearners2Goals(adaptationOptions, cl, rePL, reLA);
		}
	}

	private void doublePlLaGoals(boolean training) {
		JSONObject adjInspection = new JSONObject();
		adjInspection.put("adapIndices", new JSONArray());
		adjInspection.put("packetLoss", new JSONArray());
		adjInspection.put("energyConsumption", new JSONArray());
		adjInspection.put("latency", new JSONArray());
		adjInspection.put("regressionPLBefore", new JSONArray());
		adjInspection.put("regressionLABefore", new JSONArray());
		adjInspection.put("classificationBefore", new JSONArray());
		adjInspection.put("regressionPLAfter", new JSONArray());
		adjInspection.put("regressionLAAfter", new JSONArray());
		adjInspection.put("classificationAfter", new JSONArray());

		addPredictionsBeforeLearning(adjInspection.getJSONArray("classificationBefore"), 
			adjInspection.getJSONArray("regressionPLBefore"),
			adjInspection.getJSONArray("regressionLABefore"));


		List<Long> verifTimes = new ArrayList<>();
		// Check all the adaptation options with activFORMS
		for (AdaptationOption adaptationOption : adaptationOptions) {
			Long startTime = System.currentTimeMillis();
			smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
				adaptationOption.verificationResults);
			verifTimes.add(System.currentTimeMillis() - startTime);

			adjInspection.getJSONArray("packetLoss").put(adaptationOption.verificationResults.packetLoss);
			adjInspection.getJSONArray("energyConsumption").put(adaptationOption.verificationResults.energyConsumption);
			adjInspection.getJSONArray("latency").put(adaptationOption.verificationResults.latency);
			adjInspection.getJSONArray("adapIndices").put(adaptationOption.overallIndex);
		}

		int timeCap = ConfigLoader.getInstance().getTimeCap();

		if (training) {
			int amtOptions = adaptationOptions.size();
			long totalTime = 0;
			List<Integer> verifiedOptions = new ArrayList<>();
			
			for (int i = 0; i < amtOptions; i++) {
				int actualIndex = (i + lastLearningIndex) % amtOptions;
	
				if (totalTime / 1000 > timeCap) {
					lastLearningIndex = actualIndex;
					break;
				}
	
				totalTime += verifTimes.get(actualIndex);
				verifiedOptions.add(actualIndex);
			}
			List<AdaptationOption> options = verifiedOptions.stream().map(i -> adaptationOptions.get(i)).collect(Collectors.toList());

			// If we are training, send the entire adaptation space to the learners and check what they have learned
			send(options, TaskType.PLLAMULTICLASS, Mode.TRAINING);
			send(options, TaskType.PLLAMULTIREGR, Mode.TRAINING);

			predictionLearners2Goals(adaptationOptions, adjInspection.getJSONArray("classificationAfter"),
				adjInspection.getJSONArray("regressionPLAfter"), adjInspection.getJSONArray("regressionLAAfter"));

		} else {
			// If we are testing, send the adjustments to the learning models and check their predictions again
			// Parse the classification and regression results from the JSON responses.
			final List<Integer> classificationResults = adjInspection.getJSONArray("classificationBefore")
				.toList().stream().map(o -> Integer.parseInt(o.toString()))
				.collect(Collectors.toList());
			final List<Float> regressionResultsPL = adjInspection.getJSONArray("regressionPLBefore")
				.toList().stream().map(o -> Float.parseFloat(o.toString()))
				.collect(Collectors.toList());
			final List<Float> regressionResultsLA = adjInspection.getJSONArray("regressionLABefore")
				.toList().stream().map(o -> Float.parseFloat(o.toString()))
				.collect(Collectors.toList());

			Goal pl = Goals.getInstance().getPacketLossGoal();
			Goal la = Goals.getInstance().getLatencyGoal();
			
			// Convert the regression predictions to the classes used in classification
			List<Integer> regressionResults = new ArrayList<>();
			for (int i = 0; i < regressionResultsPL.size(); i++) {
				regressionResults.add(
					(pl.evaluate(regressionResultsPL.get(i)) ? 1 : 0) +
					(la.evaluate(regressionResultsLA.get(i)) ? 2 : 0)
				);
			}

			List<Integer> classificationIndices = getOnlineLearningIndices(classificationResults, verifTimes);
			List<Integer> regressionIndices = getOnlineLearningIndices(regressionResults, verifTimes);

			// Send the adaptation options specific to the learners back for online learning
			send(classificationIndices.stream().map(i -> adaptationOptions.get(i)).collect(Collectors.toList()), 
				TaskType.PLLAMULTICLASS, Mode.TRAINING);
			send(regressionIndices.stream().map(i -> adaptationOptions.get(i)).collect(Collectors.toList()), 
				TaskType.PLLAMULTIREGR, Mode.TRAINING);

			// Test the predictions of the learners again after online learning to track their adjustments
			predictionLearners2Goals(adaptationOptions, adjInspection.getJSONArray("classificationAfter"),
				adjInspection.getJSONArray("regressionPLAfter"), adjInspection.getJSONArray("regressionLAAfter"));
		}


		// Send the overall results to be saved on the server
		send(adjInspection, TaskType.NONE, Mode.COMPARISON);
	}

	private List<Integer> getOnlineLearningIndices(List<Integer> predictedClasses, List<Long> verificationTimes) {

		int predictionsInClass[] = new int[4];
		for (Integer pred : predictedClasses) {
			predictionsInClass[pred] += 1;
		}

		// The indices for the options of the best class predicted
		List<Integer> indicesMain = new ArrayList<>();
		// The indices for the options which are considered for exploration
		List<Integer> indicesSub = new ArrayList<>();
		// The remaining indices which should not be considered
		List<Integer> remainingIndices = new ArrayList<>();

		if (predictionsInClass[3] > 0) {
			// There is at least one option which satisfies both goals
			for (int i = 0; i < predictedClasses.size(); i++) {
				int prediction = predictedClasses.get(i);
				if (prediction == 3) {
					indicesMain.add(i);
				} else if (prediction == 2 || prediction == 1) {
					indicesSub.add(i);
				} else {
					remainingIndices.add(i);
				}
			}
		} else if (predictionsInClass[2] + predictionsInClass[1] > 0) {
			// There is at least one option which satisfies one of the goals
			for (int i = 0; i < predictedClasses.size(); i++) {
				int prediction = predictedClasses.get(i);
				if (prediction == 0) {
					indicesSub.add(i);
				} else {
					indicesMain.add(i);
				}
			}
		} else {
			for (int i = 0; i < predictedClasses.size(); i++) {
				indicesMain.add(i);
			}
		}

		double explorationPercentage = ConfigLoader.getInstance().getExplorationPercentage();

		// Shuffle the main indices first (to ensure all options are reached after some time in case not all can be verified each cycle)
		Collections.shuffle(indicesMain);
		// Similar reasoning for the exploration indices
		Collections.shuffle(indicesSub);
		
		// Only select a percentage of the predictions of the other classes
		int subIndex = (int) Math.floor(indicesSub.size() * explorationPercentage);
		remainingIndices.addAll(indicesSub.subList(subIndex, indicesSub.size()));
		indicesSub = indicesSub.subList(0, subIndex);
		
		List<Integer> overallIndices = new ArrayList<>();
		overallIndices.addAll(indicesMain);
		overallIndices.addAll(indicesSub);

		
		int timeCap = ConfigLoader.getInstance().getTimeCap();
		int lastIndex = overallIndices.size();
		int totalTime = 0;

		for (int i = 0; i < overallIndices.size(); i++) {
			if (totalTime / 1000 > timeCap) {
				lastIndex = i;
				break;
			}

			totalTime += verificationTimes.get(i);
		}

		return overallIndices.subList(0, lastIndex);
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
	 * @param regrArrayPL The JSON array which will hold the regression predictions for packet loss.
	 * @param regrArrayLA The JSON array which will hold the regression predictions for latency.
	 */
	private void predictionLearners2Goals(List<AdaptationOption> adaptationOptions, JSONArray classArray, 
			JSONArray regrArrayPL, JSONArray regrArrayLA) {
		
		JSONObject classificationResponse = send(adaptationOptions, TaskType.PLLAMULTICLASS, Mode.TESTING);
		JSONObject regressionResponse = send(adaptationOptions, TaskType.PLLAMULTIREGR, Mode.TESTING);

		for (Object item : classificationResponse.getJSONArray("predictions")) {
			classArray.put(Integer.parseInt(item.toString()));
		}

		JSONArray pred_pl = regressionResponse.getJSONArray("predictions_pl");
		JSONArray pred_la = regressionResponse.getJSONArray("predictions_la");

		for (int i = 0; i < pred_pl.length(); i++) {
			regrArrayPL.put(Float.parseFloat(pred_pl.get(i).toString()));
			regrArrayLA.put(Float.parseFloat(pred_la.get(i).toString()));
		}
	}

}
