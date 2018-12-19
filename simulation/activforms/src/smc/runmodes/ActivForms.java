
package smc.runmodes;

import java.io.File;
import java.io.FileWriter;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.json.JSONArray;
import org.json.JSONObject;

import mapek.AdaptationOption;
import mapek.Link;
import mapek.Mote;
import mapek.TrafficProbability;
import util.ConfigLoader;


public class ActivForms extends SMCConnector {

	// for collecting raw data with the activform mode
	private JSONObject rawData;
	private JSONArray features;
	private JSONArray targets;

	public ActivForms() {
		rawData = new JSONObject();
		features = new JSONArray();
		targets = new JSONArray();
		rawData.put("features", features);
		rawData.put("targets", targets);
	}

	@Override
	public void startVerification() {
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

}
