package mapek;

//This class contains represents the QoS of something, I presume it will be 
// of a mote, but it could also be of the entire networkk
public class Qualities {
	public double packetLoss;
	public double energyConsumption;
	public double latency;

	public Qualities getCopy() {
		Qualities qualities = new Qualities();
		qualities.packetLoss = this.packetLoss;
		qualities.energyConsumption = this.energyConsumption;
		qualities.latency = this.energyConsumption;
		return qualities;
	}
}
