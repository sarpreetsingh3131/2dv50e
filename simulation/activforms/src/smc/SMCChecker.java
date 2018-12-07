/*
 * This is the ActivFORMS model checker.
 */

package smc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import mapek.Qualities;

public class SMCChecker {

	String configFilePath;

	public static String command = Paths
			.get(System.getProperty("user.dir"), "uppaal-verifyta", "verifyta -a %f -E %f -u %s").toString();


	// TODO: remove this and make use of the configLoader
	public static String DEFAULT_CONFIG_FILE_PATH = Paths.get(System.getProperty("user.dir"), "SMCConfig.properties")
			.toString();

	SMCModelLoader modelLoader;

	// The models send to the binary
	List<SMCModel> models;

	public SMCChecker(String configPath) {
		this.configFilePath = configPath;
		modelLoader = new SMCModelLoader(configPath);
	}

	public SMCChecker() {
		this(DEFAULT_CONFIG_FILE_PATH);
	}

	ExecutorService cachedPool = Executors.newCachedThreadPool();

	public List<SMCModel> getModels() {
		return models;
	}

	/*
	 * FIXME: remove
	 * Reference: http://www.mkyong.com/java/how-to-execute-shell-command-from-java/
	 */
	@SuppressWarnings("unused")
	private static String executeCommand(String command) {

		StringBuffer output = new StringBuffer();

		Process p;
		try {

			p = Runtime.getRuntime().exec(command);
			p.waitFor();
			BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
			BufferedReader errorReader = new BufferedReader(new InputStreamReader(p.getErrorStream()));

			String line = "";
			while ((line = reader.readLine()) != null) {
				output.append(line + "\n");
			}

			if (output.length() == 0) {
				while ((line = errorReader.readLine()) != null) {
					output.append(line + "\n");
				}
			}

		} catch (Exception e) {
			e.printStackTrace(System.out);
		}

		return output.toString();
	}

	public static String getCommand(String modelPath, double alpha, double epsilon) {
		String cmd = String.format(command, alpha, epsilon, modelPath);
		return cmd;
	}

	static int getRuns(String string) {
		if (string.contains(" runs)")) {
			int end = string.indexOf(" runs)");
			int start = string.lastIndexOf("(", end) + 1;
			String runs = string.substring(start, end);
			return Integer.parseInt(runs);
		} else {
			throw new RuntimeException("Couldn't parse probability");
		}
	}

	/*
	 * This method parse the following line -- States explored : 147698 states
	 */
	final static String STATES_EXPLORED = "-- States explored : ";

	static long getStatesExplored(String string) {

		int start = string.indexOf(STATES_EXPLORED);
		if (start != -1) {
			start += STATES_EXPLORED.length();
			int end = string.indexOf(" states", start);
			String runs = string.substring(start, end);
			return Long.parseLong(runs);
		} else {
			throw new RuntimeException("Couldn't parse states explored");
		}
	}

	/*
	 * This method parse the following line -- CPU user time used : 571 ms
	 */
	final static String CPU_TIME = "-- CPU user time used : ";

	static long getCPUtime(String string) {

		int start = string.indexOf(CPU_TIME);
		if (start != -1) {
			start += CPU_TIME.length();
			int end = string.indexOf(" ms", start);
			String runs = string.substring(start, end);
			return Long.parseLong(runs);
		} else {
			throw new RuntimeException("Couldn't parse CPU time");
		}
	}

	/*
	 * This method parse the following line -- Resident memory used : 6164 KiB
	 */
	final static String RESIDENT_MEM = "-- Resident memory used : ";

	static long getResidentMem(String string) {

		int start = string.indexOf(RESIDENT_MEM);
		if (start != -1) {
			start += RESIDENT_MEM.length();
			int end = string.indexOf(" KiB", start);
			String runs = string.substring(start, end);
			return Long.parseLong(runs);
		} else {
			throw new RuntimeException("Couldn't parse resident memory");
		}
	}

	static double getSimulatedValue(String string) {
		// find the last pair
		int startingIndex = string.lastIndexOf("(") + 1;
		int endIndex = string.lastIndexOf(")");
		String pair = string.substring(startingIndex, endIndex);
		String[] splitPair = pair.split(",");
		double value = Double.parseDouble(splitPair[1]);
		return value;
	}

	/**
	 * Change the configuration in the quality model (string previously read from file).
	 * @param file the content of the quality model file.
	 * @param cao the system (motes and their links, loads, ...).
	 * @param env the environment (moteloads, SNR's of links).
	 * @return the quality model with changed configuration.
	 */
	static String changeCAO(String file, String cao, String env) {

		String startText = "//&lt;Configuration&gt;";
		String endText = "//&lt;/Configuration&gt;";
		String newText = String.format("%s\n%s ", cao, env);
		file = changeText(file, startText, endText, newText);
		return file;
	}

	
	static String changeCAO(String file, @SuppressWarnings("rawtypes") HashMap cao) {

		String startText = "//&lt;Configuration&amp;gt;\n";
		String endText = "//&amp;lt;/Configuration&amp;gt;";

		String newText = String.format("\nconst ManagedSystem deltaIoT =%s;\n", cao);
		file = changeText(file, startText, endText, newText);
		return file;
	}

	static String changeText(String file, String startString, String endString, String newText) {

		if (file.contains(startString)) {
			int startIndex = file.indexOf(startString) + startString.length();
			int endIndex = file.indexOf(endString);
			String oldText = file.substring(startIndex, endIndex);
			file = file.replace(oldText, newText);
			return file;
		} else {
			throw new RuntimeException("StartString:" + startString + " not found!");
		}
	}


	public void checkCAO(String adaptationOption, String environment, Qualities verificationResults) {

		// loads and updates the models and their values specified in the SMCConfig.properties
		setInitialData(adaptationOption, environment, verificationResults);

		
		// Add commands to verify all the different quality models
		LinkedList<ExecuteCommand> commands = new LinkedList<ExecuteCommand>();
		for (SMCModel model : models) {
			String command = getCommand(model.getPath(), model.alpha, model.epsilon);
			commands.add(new ExecuteCommand(command, model));
		}

		String[] values;
		try {
			cachedPool.invokeAll(commands);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		// Collection of all the results from the model verifications
		for (ExecuteCommand command : commands) {
			// Get the part of the output from the command that contains the results of the verifier
			values = command.getResult().split("Verifying formula ");

			ModelType simType = command.getModel().getType();
			String quality = command.getModel().getKey();

			// Retrieve the results for the different qualities, dependent on the model type (simulation or probability)
			switch (quality) {
				case "latency":
					// In case of latency, the returned value for simulation is still a percentage
					verificationResults.latency = simType == ModelType.SIMULATION ?
						getSimulatedValue(values[1]) * 100 : getProbability(values[1]) * 100;
					break;
				case "energyConsumption":
					verificationResults.energyConsumption = simType == ModelType.SIMULATION ?
						getSimulatedValue(values[1]) : getProbability(values[1]) * 100;
					break;
				case "packetLoss":
					verificationResults.packetLoss = simType == ModelType.SIMULATION ?
						getSimulatedValue(values[1]) : getProbability(values[1]) * 100;
					break;
			}
		}
	}

	public void setInitialData(String cao, String env, Qualities verificationResults) {
		// cao is the adaption option
		models = new LinkedList<>();
		try {

			List<SMCModel> modelsLoadedFromProperties =  modelLoader.loadModels();

			for (SMCModel model : modelsLoadedFromProperties) {

				// updates the model to include 
				// some information about the adaption option and the environment (noise and load).
				if (model.getKey().equals("packetLoss") || model.getKey().equals("energyConsumption") || model.getKey().equals("latency")) {
					String updatedModel = changeCAO(model.getModel(), cao, env);
					Files.write(Paths.get(model.getPath()), updatedModel.getBytes(Charset.defaultCharset()));
				}
				
				// add the updated form of the model to the models
				models.add(model);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static String setSimulations(String model, int simulations) {
		String str = "simulate 1[";
		if (model.contains(str)) {
			int start = model.lastIndexOf(str) + str.length();
			int end = model.indexOf("]", start);
			String s = model.substring(start, end);
			model = model.replace(s, "&lt;=" + simulations);
		}
		return model;
	}

	static double getProbability(String string) {
		String proability = getProbabilityBounds(string);
		String[] strBounds = proability.split(",");
		Double[] bounds = new Double[] { Double.parseDouble(strBounds[0]), Double.parseDouble(strBounds[1]) };

		double avg = ((bounds[0] + bounds[1]) / 2);
		return avg;
	}

	static String getProbabilityBounds(String string) {
		if (string.contains("Pr(<> ...) in [")) {
			int index = string.indexOf("Pr(<> ...) in [") + "Pr(<> ...) in [".length();
			String proability = string.substring(index, string.indexOf("]", index));

			return proability;
		} else {
			throw new RuntimeException("Couldn't parse probability");
		}
	}
}
