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

	Probe probe;
	Effector effector;

	// Knowledge
	public static final int DISTRIBUTION_GAP = ConfigLoader.getInstance().getDistributionGap();
	Configuration currentConfiguration;
	Configuration previousConfiguration;
	List<PlanningStep> steps = new LinkedList<>();
	List<SNREquation> snrEquations = new LinkedList<>();
	List<AdaptationOption> adaptationOptions = new LinkedList<>();
	List<AdaptationOption> verifiedOptions;
	SMCConnector smcConnector = new SMCConnector();

	public FeedbackLoop() {}

	public void setProbe(Probe probe) {
		this.probe = probe;
	}

	public void setEffector(Effector effector) {
		this.effector = effector;
	}

	public void setEquations(List<SNREquation> equations) {
		snrEquations = equations;
	}

	public void start() {
		System.out.println("Feedback loop started.");

		// Run the mape-k loop and simulator for the specified amount of cycles
		for (int i = 1; i <= ConfigLoader.getInstance().getAmountOfCycles(); i++) {
			System.out.print(i + ";" + System.currentTimeMillis());
			monitor();
		}
	}

	void monitor() {
		// The method "probe.getAllMotes()" also makes sure the simulator is run for a single cycle
		ArrayList<deltaiot.services.Mote> motes = probe.getAllMotes();
		List<Mote> newMotes = new LinkedList<>();

		previousConfiguration = currentConfiguration;
		currentConfiguration = new Configuration();

		Mote newMote;
		Link newLink;
		for (deltaiot.services.Mote mote : motes) {
			newMote = new Mote();
			newMote.moteId = mote.getMoteid();
			newMote.energyLevel = mote.getBattery();
			currentConfiguration.environment.motesLoad
					.add(new TrafficProbability(mote.getMoteid(), mote.getDataProbability()));
			for (deltaiot.services.Link link : mote.getLinks()) {
				newLink = new Link();
				newLink.source = link.getSource();
				newLink.destination = link.getDest();
				newLink.distribution = link.getDistribution();
				newLink.power = link.getPower();
				newMote.links.add(newLink);
				currentConfiguration.environment.linksSNR.add(new SNR(link.getSource(), link.getDest(), link.getSNR()));
			}
			newMotes.add(newMote);
		}
		currentConfiguration.system = new ManagedSystem(newMotes);
		QoS qos = probe.getNetworkQoS(1).get(0);
		currentConfiguration.qualities.packetLoss = qos.getPacketLoss();
		currentConfiguration.qualities.energyConsumption = qos.getEnergyConsumption();

		analysis();
	}

	void analysis() {
		// analyze all link settings
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
