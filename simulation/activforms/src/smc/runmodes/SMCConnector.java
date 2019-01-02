/*
* Predicts adaption options which fullfill goal(s).
* This is called in the end of the analysis function
* It connects the mapek loop to the model checker and the learner
*/


package smc.runmodes;

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
import mapek.Goal;
import mapek.Goals;
import smc.FeatureSelection;
import smc.SMCChecker;
import util.ConfigLoader;



// TODO: make it able to abort the whole proces when you receive an error from the ML
// It will make debugging much easier.

abstract public class SMCConnector {

	List<AdaptationOption> adaptationOptions;
	Environment environment;
	SMCChecker smcChecker = new SMCChecker();
	Goals goals = Goals.getInstance();
	List<AdaptationOption> verifiedOptions;
	FeatureSelection featureSelection;

	final int TRAINING_CYCLE = ConfigLoader.getInstance().getAmountOfLearningCycles();
	int cycles = 1;
	TaskType taskType;



	public enum Mode {
		MACHINELEARNING("machinelearning"),
		ACTIVFORM("activform"), 
		COMPARISON("comparison"),
		// The new mladjustment mode is similar to the comparison mode.
		// The difference between the two is that mladjustment also checks for
		// the adjustments made to the learners after online learning every cycle.
		// This data is then sent to the python server to be saved in an output file.
		MLADJUSTMENT("mladjustment"),

		// NOTE: training and testing should only be used internally
		TRAINING("training"), 
		TESTING("testing");

		public String val;

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
		PLLAMULTICLASS("pllaclassification"),
		PLLAMULTIREGR("pllaregression"),
		// None is used in case no learning task needs to be performed
		// by the learners at the server side (e.g. when just saving data).
		NONE("none");
		
		public String val;

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


	protected SMCConnector() {
		// Load the configurations specified in the properties file (mode and tasktype)
		ConfigLoader configLoader = ConfigLoader.getInstance();
		taskType = configLoader.getTaskType();
		featureSelection = new FeatureSelection();
	}


	public void setAdaptationOptions(List<AdaptationOption> adaptationOptions, Environment environment) {
		this.adaptationOptions = adaptationOptions;
		this.environment = environment;
	}

	public void verify() {
		startVerification();
		cycles++;
	}

	abstract public void startVerification();



	/**
	 * Prepare the adaptation options (their features and targets) for the machine learner.
	 * @param adaptationOptions the options which should be prepared.
	 * @param task the task type (which is necessary to decide the target).
	 * @return a JSONObject which contains the data that should be sent to the learner.
	 */
	JSONObject parse(List<AdaptationOption> adaptationOptions, TaskType task) {
		
		JSONObject dataset = new JSONObject();
		JSONArray features = new JSONArray();
		JSONArray target = new JSONArray();

		dataset.put("features", features);
		dataset.put("target", target);
		
		Goal pl = goals.getPacketLossGoal();

		for (AdaptationOption adaptationOption : adaptationOptions) {
			// Decide the target for the adaptation option (dependent on the task type)
			if (task == TaskType.CLASSIFICATION) {
				target.put(pl.evaluate(adaptationOption.verificationResults.packetLoss) ? 1 : 0);
			} else if (task == TaskType.REGRESSION) {
				target.put((int) adaptationOption.verificationResults.packetLoss);
			} else if(task == TaskType.PLLAMULTICLASS) {
				// Makes corresponding classes for multiclass verification
				// 0 - no goals are met
				// 1 - packet loss goal is met
				// 2 - latency goal is met
				// 3 - both goals are met
				int APClass = 0;

				Goal la = goals.getLatencyGoal();

				if (pl.evaluate(adaptationOption.verificationResults.packetLoss)) APClass += 1;
				if (la.evaluate(adaptationOption.verificationResults.latency)) APClass += 2;

				target.put(APClass);
			}
		
			features.put(featureSelection.selectFeatures(adaptationOption, environment));
		}

		// Returns the dataset with the features and targets
		return dataset;
	}


	/**
	 * This method parses the adaptation options to a JSONObject and sends them to the webserver which runs the machine learner.
	 * See {@link #send(JSONObject, String, String) send(JSONObject, String, String)} for more information.
	 */
	protected JSONObject send(List<AdaptationOption> adaptationOptions, TaskType taskType, Mode mode) {
		return send(parse(adaptationOptions, taskType), taskType, mode);
	}


	/**
	 * See {@link #send(JSONObject, String, String) send(JSONObject, String, String)} for more information.
	 */
	protected JSONObject send(JSONObject dataset, TaskType taskType, Mode mode) {
		return send(dataset, taskType.val, mode.val);
	}


	/**
	 * Sends the provided dataset to the webserver running the machine learner.
	 * @param dataset The data set which should be sent over to the server.
	 * @param taskType The task that needs to be performed (e.g. testing).
	 * @param mode The mode associated with the task (e.g. classification).
	 * @return The response from the server.
	 */
	protected JSONObject send(JSONObject dataset, String taskType, String mode) {
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


