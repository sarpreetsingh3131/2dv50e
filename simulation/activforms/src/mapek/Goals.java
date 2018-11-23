package mapek;

//TODO: merge or change and add this to my class
//TODO: hardcoded goals, I will have to change this or it wont look at the other goals.
public class Goals {


	// only returns true if the second argument has a lower energy consumption then the first
	// this is used when iterating throug all verified/predicted adaption options to 
	// change the best option to the second one if the energy consumption is lower
	static public boolean optimizationGoalEnergyCosnumption(AdaptationOption bestAdaptationOption,
			AdaptationOption adaptationOption) {
		if (bestAdaptationOption == null && adaptationOption != null)
			return true;
		return adaptationOption.verificationResults.energyConsumption < bestAdaptationOption.verificationResults.energyConsumption;
	}

	// does it satify his @#$!# hardcoded goal
	static public boolean satisfyGoalPacketLoss(AdaptationOption adaptationOption) {
		return adaptationOption.verificationResults.packetLoss < 10;
	}

	// only returns true if the second argument has a lower packet loss then the first
	// this is used when iterating throug all verified/predicted adaption options to 
	// change the best option to the second one if the packet loss is lower
	static public boolean optimizationGoalPacketLoss(AdaptationOption bestAdaptationOption,
			AdaptationOption adaptationOption) {
		return adaptationOption.verificationResults.packetLoss <= bestAdaptationOption.verificationResults.packetLoss;
	}
}
