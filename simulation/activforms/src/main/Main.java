package main;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import deltaiot.client.Effector;
import deltaiot.client.Probe;
import deltaiot.client.SimulationClient;
//import deltaiot.client.SimulationClient;
import deltaiot.services.QoS;
import mapek.FeedbackLoop;
import mapek.SNREquation;
import simulator.Simulator;
import util.ConfigLoader;
import domain.*;

public class Main {

	Probe probe;
	Effector effector;
	Simulator simulator;

	public void start() {
		new Thread(() -> {
			List<SNREquation> equations = new ArrayList<>();
			List<Link> links = simulator.getMotes().stream()
				.map(Mote::getLinks)
				.flatMap(List::stream)
				.collect(Collectors.toList());

			for (Link link : links) {
				equations.add(new SNREquation(link.getFrom().getId(),
					link.getTo().getId(),
					link.getSnrEquation().multiplier,
					link.getSnrEquation().constant));
			}

			FeedbackLoop feedbackLoop = new FeedbackLoop();
			feedbackLoop.setProbe(probe);
			feedbackLoop.setEffector(effector);
			feedbackLoop.setEquations(equations);

			// StartFeedback loop
			feedbackLoop.start();

			// See results
			printResults();
	}).start();
	}

	void printResults() {
		// Get QoS data of previous runs
		// probe.getNetworkQoS() should not have less number than the number of times
		// feedback loop will run, e.g, feedback loop runs 5 times, this should have >=5
		List<QoS> qosList = probe.getNetworkQoS(ConfigLoader.getInstance().getAmountOfCycles());
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
	}

	public Simulator getSimulator() {
		return simulator;
	}
}
