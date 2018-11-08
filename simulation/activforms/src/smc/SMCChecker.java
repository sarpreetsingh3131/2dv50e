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
	 * This is the same as the call function in the ExecuteCommand.java
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
		// System.out.println(cmd);
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


	//TODO: VERY IMPORTANT FUNCTION
	// so this should be the model checker
	public void checkCAO(String adaptationOption, String environment, Qualities verificationResults) {

		setInitialData(adaptationOption, environment, verificationResults);

		LinkedList<ExecuteCommand> commands = new LinkedList<ExecuteCommand>();

		for (SMCModel model : models) {
			String command = getCommand(model.getPath(), model.alpha, model.epsilon);

			// this immediatly also triggers the call() function
			commands.add(new ExecuteCommand(command, model));
		}
		String[] values;
		double value = 0;

		try {
			cachedPool.invokeAll(commands);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		// collecting results

		//TODO: add the latency model here somehow

		for (ExecuteCommand command : commands) {

			values = command.getResult().split("Verifying formula ");
			value = 0;
			if (command.getModel().getType() == ModelType.SIMULATION) {
				value = getSimulatedValue(values[1]);
				verificationResults.energyConsumption = value;
			} else if (command.getModel().getType() == ModelType.PROBABILITY) {
				value = getProbability(values[1]);
				verificationResults.packetLoss = value * 100;
			}
			// System.out.print(value + ",");
			// value = Double.parseDouble(String.format("%.2f", value));
			// verificationResults.put(command.getModel().getKey(), value);
		}
		// System.out.println();
	}

	public void setInitialData(String cao, String env, Qualities verificationResults) {
		models = new LinkedList<>();
		try {
			for (SMCModel model : modelLoader.loadModels()) {

				//TODO: would it be oke to just add here model.getKey().equals("LatencyWithoutPackets")? 
				// plus the todo above
				if (model.getKey().equals("packetLoss") || model.getKey().equals("energyConsumption")) {
					String updatedModel = changeCAO(model.getModel(), cao, env);
					Files.write(Paths.get(model.getPath()), updatedModel.getBytes(Charset.defaultCharset()));
					models.add(model);
				}
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
