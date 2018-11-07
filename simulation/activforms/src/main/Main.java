package main;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import deltaiot.client.Effector;
import deltaiot.client.Probe;
import deltaiot.client.SimulationClient;
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

		// This starts a new thread where the main object is executed on.
		new Thread(() -> {

			// This is what effectively gets executed when you start the main
			// This is the real shit that mathers

			//TODO: I pressume this get the SNR equiations.
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

			// Start a new feedback loop
			FeedbackLoop feedbackLoop = new FeedbackLoop();
			feedbackLoop.setProbe(probe);
			feedbackLoop.setEffector(effector);
			feedbackLoop.setEquations(equations);

			// StartFeedback loop
			feedbackLoop.start();

			// See results
			printResults();

	}).start();
	// Connect probe and effectors with feedback loop
	}

	//TODO: what is this data dump exactly?
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

	//DIT IS DE JAVA MAIN DIE GESTART WORD
	public static void main(String[] args) {

		// Creeer de main die zal worden uitgevoerd
		Main ddaptation = new Main();

		// Init de simulator. Dit start de simulator objecten en geeft het
		// main object een probe en effector door
		ddaptation.initializeSimulator();

		// Je start de main in een nieuwe thread (zie hierboven)
		ddaptation.start();
	}

	//Initialises a new simulator and probe
	public void initializeSimulator() {

		// Start a completely new sim
		SimulationClient client = new SimulationClient();

		// assign a new probe, effector and simulator to the main object.
		// Variabelen direct aangesproken
		probe = client.getProbe();
		effector = client.getEffector();
		simulator = client.getSimulator();
	}

	public Simulator getSimulator() {
		return simulator;
	}
}
