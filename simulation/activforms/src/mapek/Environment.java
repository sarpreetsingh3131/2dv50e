package mapek;

import java.util.LinkedList;
import java.util.List;

import com.google.gson.Gson;

public class Environment {

	// The environmnt class seems to hold the uncertainties that impact the network performance
	// such as the noise on a link (linkSNR) and the load

	public List<SNR> linksSNR = new LinkedList<>();
	public List<TrafficProbability> motesLoad = new LinkedList<>();

	// Returns the current SNR on the link
	public double getSNR(Link link) {
		for (SNR snr : linksSNR) {
			if (snr.source == link.getSource() && link.destination == link.getDestination()) {
				return snr.SNR;
			}
		}
		throw new RuntimeException("SNR not found of the given link:" + link);
	}

	// Sets the SNR of link to newSNR
	public void setSNR(Link link, double newSNR) {
		for (SNR snr : linksSNR) {
			if (snr.source == link.getSource() && link.destination == link.getDestination()) {
				snr.SNR = newSNR;
				return;
			}
		}
		throw new RuntimeException("SNR not found of the given link:" + link);
	}

	//Returns a JSON object representing the object
	@Override
	public String toString() {
		Gson gsn = new Gson();
		return gsn.toJson(this);
	}

	// Returns a gopy of the object
	public Environment getCopy() {
		Environment env = new Environment();
		for (SNR snr : linksSNR) {
			env.linksSNR.add(snr.getCopy());
		}

		for (TrafficProbability load : motesLoad) {
			env.motesLoad.add(load.getCopy());
		}
		return env;
	}

	// Object to string.
	public String toModelString() {
		StringBuilder string = new StringBuilder();

		string.append("Environment environment = {\n{");

		for (SNR snr : linksSNR) {
			string.append(String.format("{%d, %d, %d},", snr.source, snr.destination, Math.round(snr.SNR)));
		}

		string.setLength(string.length() - 1);

		string.append("},\n{");

		for (TrafficProbability load : motesLoad) {
			string.append(String.format("{%d, %d},", load.moteId, Math.round(load.load)));
		}

		string.setLength(string.length() - 1);

		string.append("}};");

		return string.toString();
	}
}
