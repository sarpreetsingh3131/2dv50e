package smc;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;

public class SMCModelLoader {

	List<SMCModel> models;
	long lastModified;
	String configPath;

	public SMCModelLoader(String configPath) {
		this.configPath = configPath;
	}

	public List<SMCModel> loadModels() {

		Properties prop = new Properties();
		InputStream input = null;

		try {

			// TODO adjust this part to use the ConfigLoader class
			File configFile = new File(configPath);
			if (!configFile.exists()) {
				throw new RuntimeException("SMCConfig.properties file not found at following path:" + configPath);
			} else {
				long lastModified = configFile.lastModified();
				if (this.lastModified == lastModified) {
					return models;
				}
				this.lastModified = lastModified;
			}

			models = new LinkedList<>();

			input = new FileInputStream(configPath);

			// load a properties file
			prop.load(input);

			// get the property value
			// load the requirements, aka the models to be predicted by the smc
			String reqs[] = prop.getProperty("requirements").split(",");

			// get where the folders are located
			String modelsFolderName = prop.getProperty("modelsFolderName");

			//HARDCODED path...
			//String modelsFolderPath = System.getProperty("user.dir") + "/" + modelsFolderName;
			Path mfp = Paths.get(System.getProperty("user.dir"), modelsFolderName);
			String modelsFolderPath = mfp.toString();
			
			// for all models to be predicted
			for (String req : reqs) {


				String key, path, simulations = "25", alpha = "0.05", epsilon = "0.05", model;
				
				ModelType type = null;
				
				key = req.trim();
				
				// Get path to model
				String modelFileName = prop.getProperty(key + ".modelFileName");
				mfp = Paths.get(modelsFolderPath, modelFileName);
				
				// In case the target folder does not exist yet, create it
				new File(Paths.get(modelsFolderPath, "target").toString()).mkdir();

				// Copy the model to the target folder so that it is only modified there
				Files.copy(mfp, Paths.get(modelsFolderPath, "target", modelFileName), StandardCopyOption.REPLACE_EXISTING);
				
				path = Paths.get(modelsFolderPath, "target", modelFileName).toString();

				// this and the if else statement gets info on how to execute the model from the properties file
				String modelType = prop.getProperty(key + ".type");


				if (modelType.equalsIgnoreCase("simulation")) {
					type = ModelType.SIMULATION;
					simulations = prop.getProperty(key + ".totalSimulations");
				} 
				else if (modelType.equalsIgnoreCase("probability")) {
					type = ModelType.PROBABILITY;
					alpha = prop.getProperty(key + ".alpha");
					epsilon = prop.getProperty(key + ".epsilon");
				}

				// read in the model from the filepath
				model = new String(Files.readAllBytes(Paths.get(path)), Charset.defaultCharset());

				// construct model from the properties you have read
				SMCModel smcModel = new SMCModel(key, path, type, Integer.parseInt(simulations),
						Double.parseDouble(alpha), Double.parseDouble(epsilon), model);

				// add the model
				models.add(smcModel);
			}

		} catch (IOException ex) {
			ex.printStackTrace();
		} finally {
			if (input != null) {
				try {
					input.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return models;
	}
}
