package main;

import java.util.List;

import deltaiot.client.Effector;
import deltaiot.client.Probe;
import deltaiot.client.SimulationClient;
//import deltaiot.client.SimulationClient;
import deltaiot.services.QoS;
import mapek.FeedbackLoop;
import simulator.Simulator;

public class Main {

	Probe probe;
	Effector effector;
	Simulator simulator;

	public void start() {

		// get probe and effectors
		// probe = new Probe();
		// effector = new Effector();
		// SimulationClient client = new SimulationClient();
		// probe = client.getProbe();
		// effector = client.getEffector();
		new Thread(new Runnable() {

			@Override
			public void run() {
				// TODO Auto-generated method stub
				FeedbackLoop feedbackLoop = new FeedbackLoop();
				feedbackLoop.setProbe(probe);
				feedbackLoop.setEffector(effector);

				// StartFeedback loop
				feedbackLoop.start();

				// See results
				printResults();
			}
		}).start();
		// Connect probe and effectors with feedback loop
	}

	void printResults() {
		// Get QoS data of previous runs
		List<QoS> qosList = probe.getNetworkQoS(300);
		System.out.println("\nPacketLoss;EnergyConsumption");
		for (QoS qos : qosList) {
			System.out.println(String.format("%f;%f", qos.getPacketLoss(), qos.getEnergyConsumption()));
		}
	}

	public static void main(String[] args) {
		Main ddaptation = new Main();
		ddaptation.initializeSimulator();
		ddaptation.start();
	}

	public void initializeSimulator() {
		SimulationClient client = new SimulationClient();
		probe = client.getProbe();
		effector = client.getEffector();
		simulator = client.getSimulator();
		// probe = new Probe();
		// effector = new Effector();
	}

	public Simulator getSimulator() {
		return simulator;
	}
}
