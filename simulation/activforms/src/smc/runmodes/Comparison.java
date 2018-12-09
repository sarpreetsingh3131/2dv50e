
package smc.runmodes;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.json.JSONArray;
import org.json.JSONObject;

import mapek.AdaptationOption;
import mapek.Goal;


// TODO: unify comparison and mladjustment
public class Comparison extends SMCConnector {

	@Override
	public void startVerification() {
		if (cycles <= TRAINING_CYCLE) {
			training();
		} else {
			testing();
		}
	}
		
	private void training() {
		switch (taskType) {
			case CLASSIFICATION:
			case REGRESSION:
				training1Goal();
				break;
			case PLLAMULTICLASS:
				training2Goals();
				break;
			default:
				throw new RuntimeException(
					String.format("Unsupported task type for MLAdjustment: %s", taskType.val));
		}
	}
	
	private void testing() {
		switch (taskType) {
			case CLASSIFICATION:
			case REGRESSION:
				testing1Goal();
				break;
			case PLLAMULTICLASS:
				testing2Goals();
				break;
			default:
				throw new RuntimeException(
					String.format("Unsupported task type for MLAdjustment: %s", taskType.val));
		}
	}
	
	private void training1Goal() {
		int space = 0;
		Goal pl = goals.getPacketLossGoal();
	
		for (AdaptationOption adaptationOption : adaptationOptions) {
			smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
				adaptationOption.verificationResults);
			if (pl.evaluate(adaptationOption.verificationResults.packetLoss)) {
				space++;
			}
		}
		
		System.out.print(";" + space);
		send(adaptationOptions, TaskType.CLASSIFICATION, Mode.TRAINING);
		send(adaptationOptions, TaskType.REGRESSION, Mode.TRAINING);
	}

	private void training2Goals() {
		throw new RuntimeException("Training for multiple goals in comparison is unsupported at this moment.");		
	}


	private void testing1Goal() {
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


	private void testing2Goals() {
		throw new RuntimeException("Testing for multiple goals in comparison is unsupported at this moment.");		
	}
}
