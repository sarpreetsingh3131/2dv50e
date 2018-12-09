package mapek;

public class Configuration {

	// This is a class which hold all motes of the network.
	// However, the whole network can be derived from it because of the info the motes hold.
	// This is the architecture of the network allong with the current 
	// adaption options (instellingen).
	ManagedSystem system;

	// The environment contains the SNR on a given link and the load.
	Environment environment;

	// An object holding single qualities.
	Qualities qualities;

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
