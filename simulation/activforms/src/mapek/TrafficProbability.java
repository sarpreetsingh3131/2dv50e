package mapek;

public class TrafficProbability {

	// An object of this class will be an element in a list of a mote object,
	// one for every cycle to represent the load on the mote for that cycle.
	public int moteId;
	public double load;

	public TrafficProbability(int moteId, double load) {
		this.moteId = moteId;
		this.load = load;
	}

	public TrafficProbability getCopy() {
		return new TrafficProbability(moteId, load);
	}
}
