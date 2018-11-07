package mapek;

public class Configuration {

	// This is a class which hold all motes of the network.
	// However, the whole network can be derived from it because of the info the motes hold.
	// This is the architecture of the network allong with the current 
	// adaption options (instellingen).
	ManagedSystem system;

	// The environment contains the SNR on a given link and the load.
	// There is nothing done however with the load variable of the object in the class,
	// so I presume somewhere it is changed directly and not through a setter.
	Environment environment;

	// An object holding single qualities.
	// For now it only contains packetloss and energy consumption
	// and a singel value for them
	// However, it should be looked at to add latency and if it is really necessary
	// why you wouldnt just use the deltaIoT class for it
	Qualities qualities;

	// The methods speak for themselves
	public Configuration() {
		system = new ManagedSystem();
		environment = new Environment();
		qualities = new Qualities();
	}

	protected Configuration getCopy() {
		Configuration newConfiguration = new Configuration();
		newConfiguration.system = system.getCopy();
		newConfiguration.environment = environment.getCopy();
		newConfiguration.qualities = qualities.getCopy();
		return newConfiguration;
	}
}
