package mapek;

//This class contains represents the QoS of something, I presume it will be 
// of a mote, but it could also be of the entire networkk
// TODO: adapt this to also inlcude the latency and why does he not use the QoS class of deltaIoT?
public class Qualities {
	public double packetLoss;
	public double energyConsumption;

	public Qualities getCopy() {
		Qualities qualities = new Qualities();
		qualities.packetLoss = this.packetLoss;
		qualities.energyConsumption = this.energyConsumption;
		return qualities;
	}
}
