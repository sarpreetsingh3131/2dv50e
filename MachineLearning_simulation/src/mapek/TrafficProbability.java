package mapek;

public class TrafficProbability {
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
