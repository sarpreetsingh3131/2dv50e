package mapek;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import deltaiot.client.Effector;
import deltaiot.services.LinkSettings;
import deltaiot.services.QoS;
import smc.SMCConnector;
import util.ConfigLoader;
import deltaiot.client.Probe;

public class FeedbackLoop {

	//The probe and effector of the network being worked on.
	Probe probe;
	Effector effector;

	// Knowledge

	//TODO: distribution gap aanpassen om grotere space te creeeren, is enige mogelijkeid.
	public static final int DISTRIBUTION_GAP = ConfigLoader.getInstance().getDistributionGap();
	
	//Dit zullen volledige staten(=configuratie) zijn van het netwerk op een bepaald moment
	Configuration currentConfiguration;
	Configuration previousConfiguration;

	// TODO: Geen idee
	// Als ik de class bekijk lijkt het alsof steps(i)
	// de aanpassing is die je deed op het einde van cycle i+1
	// aan de power en distribution
	// Er moet echter verder inde code gekeken worden
	List<PlanningStep> steps = new LinkedList<>();
	
	//Stellen de ruis op een bepaalde link voor (op een bepaald tijdstip?)
	//Zijn ergens manueel ingetypt en komt van gemeten data van een week van het echte netwerk
	List<SNREquation> snrEquations = new LinkedList<>();

	// Adaption space
	List<AdaptationOption> adaptationOptions = new LinkedList<>();
	
	// TODO: geen idee
	List<AdaptationOption> verifiedOptions;

	// De connector die met de machine learner connecteerd.
	SMCConnector smcConnector = new SMCConnector();

	// Gets called from the main.
	public FeedbackLoop() {
		//TODO: leg me uit hoe die lamda werkt aub en waar de SNR equitions worden toegevoegd
	}

	//Sets the probe of the feedbackloop
	public void setProbe(Probe probe) {
		this.probe = probe;
	}

	//Sets the effector of the feedbackloop
	public void setEffector(Effector effector) {
		this.effector = effector;
	}

	public void setEquations(List<SNREquation> equations) {
		snrEquations = equations;
	}

	//This is were the feedback loop really starts.
	public void start() {
		System.out.println("Feedback loop started.");

		// Run the mape-k loop and simulator for the specified amount of cycles
		for (int i = 1; i <= ConfigLoader.getInstance().getAmountOfCycles(); i++) {
			System.out.print(i + ";" + System.currentTimeMillis());
			
			// Start the monitor part of the mapek loop
			// The rest of the parts are each called in the previous parts
			monitor();
		}
	}


	void monitor() {
		// The method "probe.getAllMotes()" also makes sure the simulator is run for a single cycle
		ArrayList<deltaiot.services.Mote> motes = probe.getAllMotes();
		
		
		List<Mote> newMotes = new LinkedList<>();

		// configuratie/cyclus/state opschuiven
		// CurrentCon.. is initialised as null.
		// So prevConf will be null on the first cycle
		// TODO: it isnt! Maybe Java does it for us.
		previousConfiguration = currentConfiguration;

		// Init new configuration
		currentConfiguration = new Configuration();

		// Maakt copy van netwerk in huidige staat

		// Makes copy of the IoT network in its current state
		Mote newMote;
		Link newLink;

		// Iterate through all the motes OF THE SIMULATOR
		for (deltaiot.services.Mote mote : motes) {

			// Make a new mote and give it the ID of the mote being iterated on
			newMote = new Mote();
			newMote.moteId = mote.getMoteid();

			// Adds the current battery level to the mote
			newMote.energyLevel = mote.getBattery();

			// The motesLoad is a list with the load of the motes.
			// I think every element of that list represents the load
			// on the mote in a cycle in the past.
			// Here you add a new load to the list for the current cycle
			// TODO: the way the load is calculated should be looked at.
			// TODO: the dataprobability is a constant? how can that be the load?
			currentConfiguration.environment.motesLoad
					.add(new TrafficProbability(mote.getMoteid(), mote.getDataProbability()));
			
			
			// Copy the links and its SNR
			//TODO: so the SNR and load never change for a mote throughout time?
			for (deltaiot.services.Link link : mote.getLinks()) {
				newLink = new Link();
				newLink.source = link.getSource();
				newLink.destination = link.getDest();
				newLink.distribution = link.getDistribution();
				newLink.power = link.getPower();
				newMote.links.add(newLink);
				currentConfiguration.environment.linksSNR.add(new SNR(link.getSource(), link.getDest(), link.getSNR()));
			}
			
			// add the mote to the configuration
			newMotes.add(newMote);
		}

		// This saves the architecture of the system to the new configuration by adding the 
		// new motes which contain all the necessary data
		currentConfiguration.system = new ManagedSystem(newMotes);
		
		//getNetworkQoS(n) returns a list of the QoS
		// values of the n previous cycles.
		//This returns the latest QoS and
		// returns the first (and only) element of the list.
		QoS qos = probe.getNetworkQoS(1).get(0);

		//Hier neemt hij enkel deze 2 mee TODO
		// Zie of je de latency enzo ook kunt meegeven
		// Adds the QoS of the previous configuration to the current configuration,
		// probably to pass on to the learner so he can use this to online learn
		// TODO: modify this to multiple goals
		currentConfiguration.qualities.packetLoss = qos.getPacketLoss();
		currentConfiguration.qualities.energyConsumption = qos.getEnergyConsumption();

		// Call the next step off the mapek
		analysis();
	}





	// Gets called at the end of the monitor method
	void analysis() {

		// analyze all link settings
		// returns false if no change has to be made.
		// 
		// Otherwise it returns true, to see when you should look at the definition below
		boolean adaptationRequired = analysisRequired();

		if (!adaptationRequired)
			return;

		AdaptationOption newPowerSettingsConfig = new AdaptationOption();
		newPowerSettingsConfig.system = currentConfiguration.system.getCopy();
		analyzePowerSettings(newPowerSettingsConfig);
		removePacketDuplication(newPowerSettingsConfig);
		composeAdaptationOptions(newPowerSettingsConfig);

		smcConnector.setAdaptationOptions(adaptationOptions, currentConfiguration.environment);
		smcConnector.startVerification();
		verifiedOptions = adaptationOptions;

		planning();
	}

	void composeAdaptationOptions(AdaptationOption newConfiguration) {
		List<Mote> moteOptions = new LinkedList<>();
		if (adaptationOptions.size() <= 1) {
			// adaptationOptions.add(newConfiguration);
			// generate adaptation options for the first time
			int initialValue = 0;
			for (Mote mote : newConfiguration.system.motes.values()) {
				if (mote.getLinks().size() == 2) {
					mote = mote.getCopy();
					moteOptions.clear();
					for (int i = initialValue; i <= 100; i += DISTRIBUTION_GAP) {
						mote.getLink(0).setDistribution(i);
						mote.getLink(1).setDistribution(100 - i);
						moteOptions.add(mote.getCopy());
					}
					initialValue = 20;
					saveAdaptationOptions(newConfiguration, moteOptions, mote.getMoteId());
				}
			}
		}
	}

	private void saveAdaptationOptions(AdaptationOption firstConfiguration, List<Mote> moteOptions, int moteId) {
		AdaptationOption newAdaptationOption;
		if (adaptationOptions.isEmpty()) {
			for (int j = 0; j < moteOptions.size(); j++) {
				newAdaptationOption = firstConfiguration.getCopy();
				newAdaptationOption.system.motes.put(moteId, moteOptions.get(j));
				adaptationOptions.add(newAdaptationOption);
			}
		} else {
			int size = adaptationOptions.size();
			for (int i = 0; i < size; i++) {
				for (int j = 0; j < moteOptions.size(); j++) {
					newAdaptationOption = adaptationOptions.get(i).getCopy();
					newAdaptationOption.system.motes.put(moteId, moteOptions.get(j));
					adaptationOptions.add(newAdaptationOption);
				}
			}
		}
	}

	private void analyzePowerSettings(AdaptationOption newConfiguration) {
		int powerSetting;
		double newSNR;
		for (Mote mote : newConfiguration.system.motes.values()) {
			for (Link link : mote.getLinks()) {
				powerSetting = link.getPower();
				newSNR = currentConfiguration.environment.getSNR(link);
				// find interference
				double diffSNR = getSNR(link.getSource(), link.getDestination(), powerSetting) - newSNR;
				if (powerSetting < 15 & newSNR < 0 && newSNR != -50) {
					while (powerSetting < 15 && newSNR < 0) {
						newSNR = getSNR(link.getSource(), link.getDestination(), ++powerSetting) - diffSNR;
					}
				} else if (newSNR > 0 && powerSetting > 0) {
					do {
						newSNR = getSNR(link.getSource(), link.getDestination(), powerSetting - 1) - diffSNR;
						if (newSNR >= 0) {
							powerSetting--;
						}
					} while (powerSetting > 0 && newSNR >= 0);
				}
				if (link.getPower() != powerSetting) {
					link.setPower(powerSetting);
					currentConfiguration.environment.setSNR(link,
							getSNR(link.getSource(), link.getDestination(), powerSetting) - diffSNR);
				}
			}
		}
	}

	private void removePacketDuplication(AdaptationOption newConfiguration) {
		for (Mote mote : newConfiguration.system.motes.values()) {
			if (mote.getLinks().size() == 2) {
				if (mote.getLink(0).getDistribution() == 100 && mote.getLink(1).getDistribution() == 100) {
					mote.getLink(0).setDistribution(0);
					mote.getLink(1).setDistribution(100);
				}
			}
		}
	}

	double getSNR(int source, int destination, int newPowerSetting) {
		for (SNREquation equation : snrEquations) {
			if (equation.source == source && equation.destination == destination) {
				return equation.multiplier * newPowerSetting + equation.constant;
			}
		}
		throw new RuntimeException("Link not found:" + source + "-->" + destination);
	}

	static final int SNR_BELOW_THRESHOLD = 0;
	static final int SNR_UPPER_THRESHOLD = 5;
	static final int ENERGY_CONSUMPTION_THRESHOLD = 5;
	static final int PACKET_LOSS_THRESHOLD = 5;
	static final int MOTES_TRAFFIC_THRESHOLD = 10;
	static final int MAX_LINKS = 17;
	static final int MAX_MOTES = 15;

	// int i;
	boolean analysisRequired() {
		// for simulation we use adaptation after 4 periods
		// return i++%4 == 0;

		// if first time perform adaptation
		if (previousConfiguration == null)
			return true;

		// Check LinksSNR
		double linksSNR;
		for (int j = 0; j < MAX_LINKS; j++) {
			linksSNR = currentConfiguration.environment.linksSNR.get(j).SNR;
			if (linksSNR < SNR_BELOW_THRESHOLD || linksSNR > SNR_UPPER_THRESHOLD) {
				return true;
			}
		}

		// Check MotesTraffic
		double diff;

		for (int i = 2; i <= MAX_MOTES; i++) {
			diff = currentConfiguration.environment.motesLoad.get(i).load
					- previousConfiguration.environment.motesLoad.get(i).load;
			if (diff > MOTES_TRAFFIC_THRESHOLD || diff > -MOTES_TRAFFIC_THRESHOLD) {
				return true;
			}
		}

		// check qualities
		if ((currentConfiguration.qualities.packetLoss > previousConfiguration.qualities.packetLoss
				+ PACKET_LOSS_THRESHOLD)
				|| (currentConfiguration.qualities.energyConsumption > previousConfiguration.qualities.energyConsumption
						+ ENERGY_CONSUMPTION_THRESHOLD)) {
			return true;
		}

		// check if system settings are not what should be
		return !currentConfiguration.system.toString().equals(previousConfiguration.system.toString());

	}


	void planning() {
		AdaptationOption bestAdaptationOption = null;

		for (int i = 0; i < verifiedOptions.size(); i++) {
			if (Goals.satisfyGoalPacketLoss(verifiedOptions.get(i))
					&& Goals.optimizationGoalEnergyCosnumption(bestAdaptationOption, verifiedOptions.get(i))) {

				bestAdaptationOption = verifiedOptions.get(i);
			}
		}

		if (bestAdaptationOption == null) {
			// System.out.println("Using faile safety configuration");

			for (int i = 0; i < verifiedOptions.size(); i++) {
				if (Goals.optimizationGoalEnergyCosnumption(bestAdaptationOption, verifiedOptions.get(i))) {
					bestAdaptationOption = verifiedOptions.get(i);
				}
			}
		}
		// System.out.print(";" + bestAdaptationOption.verificationResults.packetLoss +
		// ";"
		// + bestAdaptationOption.verificationResults.energyConsumption);
		// System.out.println("SelectedOption:" + bestAdaptationOption);

		// Go through all links
		Link newLink, oldLink;
		for (Mote mote : bestAdaptationOption.system.motes.values()) {
			for (int i = 0; i < mote.getLinks().size(); i++) {
				newLink = mote.getLinks().get(i);
				oldLink = currentConfiguration.system.motes.get(mote.moteId).getLink(i);
				if (newLink.getPower() != oldLink.getPower()) {
					steps.add(new PlanningStep(Step.CHANGE_POWER, newLink, newLink.getPower()));
				}
				if (newLink.getDistribution() != oldLink.getDistribution()) {
					steps.add(new PlanningStep(Step.CHANGE_DIST, newLink, newLink.getDistribution()));
				}
			}
		}

		if (steps.size() > 0) {
			execution();
		} else {
			System.out.println(";" + System.currentTimeMillis());
		}
	}


	void execution() {
		boolean addMote;
		List<Mote> motesEffected = new LinkedList<Mote>();
		for (Mote mote : currentConfiguration.system.motes.values()) {
			addMote = false;
			for (PlanningStep step : steps) {
				if (step.link.getSource() == mote.getMoteId()) {
					addMote = true;
					if (step.step == Step.CHANGE_POWER) {
						findLink(mote, (step.link.getDestination())).setPower(step.value);
					} else if (step.step == Step.CHANGE_DIST) {
						findLink(mote, (step.link.getDestination())).setDistribution(step.value);
					}
				}
			}
			if (addMote)
				motesEffected.add(mote);
		}
		List<LinkSettings> newSettings;

		// System.out.println("Adaptations:");
		for (Mote mote : motesEffected) {
			// printMote(mote);
			newSettings = new LinkedList<LinkSettings>();
			for (Link link : mote.getLinks()) {
				newSettings.add(newLinkSettings(mote.getMoteId(), link.getDestination(), link.getPower(),
						link.getDistribution(), 0));
			}
			effector.setMoteSettings(mote.getMoteId(), newSettings);
		}
		steps.clear();
		System.out.print(";" + System.currentTimeMillis() + "\n");
	}


	Link findLink(Mote mote, int dest) {
		for (Link link : mote.getLinks()) {
			if (link.getDestination() == dest)
				return link;
		}
		throw new RuntimeException(String.format("Link %d --> %d not found", mote.getMoteId(), dest));
	}


	public LinkSettings newLinkSettings(int src, int dest, int power, int distribution, int sf) {
		LinkSettings settings = new LinkSettings();
		settings.setSrc(src);
		settings.setDest(dest);
		settings.setPowerSettings(power);
		settings.setDistributionFactor(distribution);
		settings.setSpreadingFactor(sf);
		return settings;
	}


	void printMote(Mote mote) {
		System.out.println(String.format("MoteId: %d, BatteryRemaining: %f, Links:%s", mote.getMoteId(),
				mote.getEnergyLevel(), getLinkString(mote.getLinks())));
	}

	
	String getLinkString(List<Link> links) {
		StringBuilder strBuilder = new StringBuilder();
		for (Link link : links) {
			strBuilder.append(String.format("[Dest: %d, Power:%d, DistributionFactor:%d]", link.getDestination(),
					link.getPower(), link.getDistribution()));
		}
		return strBuilder.toString();
	}
}
