package mapek;

import com.google.gson.Gson;

public class AdaptationOption {
	public ManagedSystem system;
	public Qualities verificationResults = new Qualities();

	protected AdaptationOption getCopy() {
		AdaptationOption newOption = new AdaptationOption();
		newOption.system = system.getCopy();
		newOption.verificationResults = verificationResults.getCopy();
		return newOption;
	}

	@Override
	public String toString() {
		Gson gsn = new Gson();
		return gsn.toJson(this);
	}

	public String toModelString() {
		StringBuilder string = new StringBuilder();
		string.append("\nManagedSystem deltaIoT = {{\n");
		Mote mote;
		for (int i = 2; i <= 15; i++) {
			mote = system.getMote(i);

			string.append(mote.getModelString());
			string.append(",");
		}

		string.setLength(string.length() - 1);
		string.append("\n}};");
		return string.toString();
	}
}
