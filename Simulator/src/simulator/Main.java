package simulator;

import java.util.ArrayList;
import java.util.Random;

import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.BasicResponseHandler;
import org.apache.http.impl.client.HttpClientBuilder;
import org.json.JSONArray;
import org.json.JSONObject;

import deltaiot.DeltaIoTSimulator;
import deltaiot.client.SimulationClient;
import deltaiot.services.QoS;

public class Main {

	/*
	 * enum Type { CLASSIFICATION("classification"), REGRESSION("regression"),
	 * CLUSTERING("clustering");
	 * 
	 * String value;
	 * 
	 * Type(String value) { this.value = value; } }
	 * 
	 * enum Classification { SVC("svc");
	 * 
	 * String value;
	 * 
	 * Classification(String value) { this.value = value; } }
	 */

	public static void main(String[] args) {
		run();
	}

	public static void run() {
		ArrayList<ArrayList<deltaiot.services.Mote>> adaptations = new ArrayList<>();
		ArrayList<ArrayList<QoS>> result = new ArrayList<>();
		JSONObject data = new JSONObject();
		Random ran = new Random();
		int[] powers = new int[17];
		int[] distributions = new int[6];
		double globalInterference = -3;
		int probability1 = 0;
		int probability2 = 0;

		// for (int a = 0; a < 1000; a++)
		while (globalInterference < 10) {

			for (int b = 0; b < powers.length; b++) {
				powers[b] = ran.nextInt(101) % 16;
			}

			for (int c = 0; c < distributions.length; c++) {
				distributions[c] = c == 1 ? 100 : 0;
			}

			probability1 = ran.nextInt(101);
			probability2 = ran.nextInt(101);

			for (int i = 0; i < 6; i++) {
				distributions[2] = 0;
				distributions[3] = 100;

				for (int j = 0; j < 6; j++) {
					distributions[4] = 0;
					distributions[5] = 100;

					for (int k = 0; k < 6; k++) {
						Simulator simul = DeltaIoTSimulator.createSimulatorForTraining(distributions, powers,
								globalInterference, probability1, probability2);

						SimulationClient client = new SimulationClient(simul);
						// for (int m = 0; m < 10; m++) {
						adaptations.add(client.getProbe().getAllMotes());
						// System.out.println(adaptations.get(0));
						result.add(client.getNetworkQoS(1));
						// System.out.println(client.getNetworkQoS(1));
						// }
						distributions[4] += 20;
						distributions[5] -= 20;
					}
					distributions[2] += 20;
					distributions[3] -= 20;
				}
				distributions[0] += 20;
				distributions[1] -= 20;
			}
			globalInterference += 0.01;

			data.put("result", result);
			data.put("adaptations", adaptations);
			System.out.println(data);
			adaptations.clear();
			result.clear();
			data.remove("adaptations");
			data.remove("result");

		}
		/*
		 * Do logic for (int i = 0; i < 96; ++i) { simul.doSingleRun();
		 * simul.doMultipleRuns(96);
		 * 
		 * for(Gateway gateway: simul.getGateways()) { System.out.println(gateway); }
		 * for(Mote mote: motes) { System.out.println(mote); } }
		 * 
		 * QoS qos; for(int i = 0; i < simul.getQosValues().size(); i++){ qos =
		 * simul.getQosValues().get(i); System.out.format("%d, %f, %f, %f, %f\n", i,
		 * qos.getPacketLoss(), qos.getEnergyConsumption(), qos.getLatency(),
		 * qos.getQueueLoss()); }
		 */

	}

	/*
	 * static void sendDataToServer(ArrayList<ArrayList<deltaiot.services.Mote>>
	 * adaptations, ArrayList<ArrayList<QoS>> result, String type, String clf) {
	 * 
	 * try { JSONObject data = new JSONObject(); data.put("adaptations", new
	 * JSONArray(adaptations)); data.put("result", new JSONArray(result));
	 * 
	 * // JsonObject sendData = new JsonObject(); // sendData.add("adaptations", new
	 * Gson().toJsonTree(new // JSONArray(adaptations))); // sendData.add("result",
	 * new Gson().toJsonTree(new JSONArray(result)));
	 * 
	 * HttpClient client = HttpClientBuilder.create().build(); HttpPost http = new
	 * HttpPost("http://localhost:8000/?type=" + type + "&mode=training&clf=" +
	 * clf); http.setEntity(new StringEntity(data.toString()));
	 * http.setHeader("Content-Type", "application/json"); System.out.println(new
	 * JSONObject(client.execute(http, new BasicResponseHandler())).toString()); }
	 * catch (Exception e) { e.printStackTrace(); } }
	 */
}
