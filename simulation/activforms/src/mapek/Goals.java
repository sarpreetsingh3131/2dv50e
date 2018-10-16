package mapek;

public class Goals {

	static public boolean optimizationGoalEnergyCosnumption(AdaptationOption bestAdaptationOption,
			AdaptationOption adaptationOption) {
		if (bestAdaptationOption == null && adaptationOption != null)
			return true;
		return adaptationOption.verificationResults.energyConsumption < bestAdaptationOption.verificationResults.energyConsumption;
	}

	static public boolean satisfyGoalPacketLoss(AdaptationOption adaptationOption) {
		return adaptationOption.verificationResults.packetLoss < 10;
	}

	static public boolean optimizationGoalPacketLoss(AdaptationOption bestAdaptationOption,
			AdaptationOption adaptationOption) {
		return adaptationOption.verificationResults.packetLoss <= bestAdaptationOption.verificationResults.packetLoss;
	}
}
