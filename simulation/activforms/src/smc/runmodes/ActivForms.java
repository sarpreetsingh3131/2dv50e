
package smc.runmodes;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;

import mapek.AdaptationOption;
import mapek.Environment;
import mapek.Link;
import mapek.Mote;
import mapek.SNR;
import mapek.TrafficProbability;


public class ActivForms extends SMCConnector {

	public ActivForms() {}

	@Override
	public void startVerification() {
		System.out.print(";" + adaptationOptions.size());

		Long[] verifTimes = new Long[adaptationOptions.size()]; 
		int index = 0;

		// Check all the adaptation options with activFORMS (and keep track of the verification time of each option)
		for (AdaptationOption adaptationOption : adaptationOptions) {
			Long startTime = System.currentTimeMillis();

			smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
				adaptationOption.verificationResults);
				
			verifTimes[index] = System.currentTimeMillis() - startTime;
			index++;
		}


		storeAllFeaturesAndTargets(adaptationOptions, environment, cycles, verifTimes);

	}


	private void storeAllFeaturesAndTargets(List<AdaptationOption> adaptationOptions, Environment env, int cycle, Long[] verifTimes) {
		// Store the features and the targets in their respective files
		File feature_selection = new File(
			Paths.get(System.getProperty("user.dir"), "activforms", "log", "dataset_with_all_features" + cycle + ".json").toString());

		if (feature_selection.exists()) {
			// At the first cycle, remove the file if it already exists
			feature_selection.delete();
		}
		
		try {
			feature_selection.createNewFile();
			JSONObject root = new JSONObject();
			root.put("verification_times", new JSONArray());
			root.put("features", new JSONArray());
			root.put("target_classification_packetloss", new JSONArray());
			root.put("target_regression_packetloss", new JSONArray());
			root.put("target_classification_latency", new JSONArray());
			root.put("target_regression_latency", new JSONArray());
			root.put("target_regression_energyconsumption", new JSONArray());
			FileWriter writer = new FileWriter(feature_selection);
			writer.write(root.toString(2));
			writer.close();
		} catch (IOException e) {
			throw new RuntimeException(
				String.format("Could not create the output file at %s", feature_selection.toPath().toString()));
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
				root.getJSONArray("target_regression_packetloss").put(option.verificationResults.packetLoss);
				
				// Latency values
				root.getJSONArray("target_classification_latency").put(
					goals.getLatencyGoal().evaluate(option.verificationResults.latency) ? 1 : 0);
				root.getJSONArray("target_regression_latency").put(option.verificationResults.latency);

				// Energy consumption values
				root.getJSONArray("target_regression_energyconsumption").put(option.verificationResults.energyConsumption);
			}

			for (Long verifTime : verifTimes) {
				root.getJSONArray("verification_times").put(verifTime);
			}

			FileWriter writer = new FileWriter(feature_selection);
			writer.write(root.toString(1));
			writer.close();
			
		} catch (IOException e) {
			throw new RuntimeException(
				String.format("Could not write to the output file at %s", feature_selection.toPath().toString()));
		}
	}

}
