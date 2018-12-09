package smc;

import java.util.Arrays;
import java.util.List;

import org.json.JSONArray;

import mapek.AdaptationOption;
import mapek.Environment;
import mapek.Link;
import mapek.Mote;
import mapek.SNR;
import mapek.TrafficProbability;
import util.ConfigLoader;
import util.Pair;

public class FeatureSelection {
    private String network;
    private List<Pair<Integer,Integer>> selectedSNR_DeltaIoTv1;
    private List<Pair<Integer,Integer>> selectedPower_DeltaIoTv1;
    private List<Pair<Integer,Integer>> selectedDist_DeltaIoTv1;
    private List<Integer> selectedLoad_DeltaIoTv1;
    
    public FeatureSelection() {
        network = ConfigLoader.getInstance().getSimulationNetwork();

        // Initialisation of selected features for DeltaIoTv1
        selectedSNR_DeltaIoTv1 = Arrays.asList(
            new Pair<>(2,4),
            new Pair<>(4,1),
            new Pair<>(5,9),
            new Pair<>(6,4),
            new Pair<>(7,3),
            new Pair<>(10,5),new Pair<>(10,6),
            new Pair<>(12,3),new Pair<>(12,7),
            new Pair<>(14,12),
            new Pair<>(15,12)
        );
        selectedPower_DeltaIoTv1 = Arrays.asList(
            new Pair<>(3,1),
            new Pair<>(7,2),
            new Pair<>(8,1),
            new Pair<>(9,1),
            new Pair<>(10,5),new Pair<>(10,6),
            new Pair<>(11,7),
            new Pair<>(13,11)
        );
        selectedDist_DeltaIoTv1 = Arrays.asList(
            new Pair<>(7,2),new Pair<>(7,3),
            new Pair<>(10,5),new Pair<>(10,6),
            new Pair<>(12,3),new Pair<>(12,7)
        );
        selectedLoad_DeltaIoTv1 = Arrays.asList(10, 13);
    }

    public JSONArray selectFeatures(AdaptationOption option, Environment env) {
        switch (network) {
            case "DeltaIoTv1":
                return selectFeaturesDeltaIoTv1(option, env);
            case "DeltaIoTv2":
                return selectFeaturesDeltaIoTv2(option, env);
        }
        throw new RuntimeException(String.format("Unsupported network for feature selection: %s", network));
    }

    public JSONArray selectFeaturesDeltaIoTv1(AdaptationOption option, Environment env) {
        JSONArray features = new JSONArray();
        
        // Add the SNR values of certain links in the environment
        for (SNR snr : env.linksSNR) {
            if (selectedSNR_DeltaIoTv1.stream()
                .anyMatch(l -> l.first == snr.source && l.second == snr.destination)) {
                features.put((int) snr.SNR);
            }
        }
        
        // Add the power settings for certain links
        for (Mote mote : option.system.motes.values()) {
            for (Link link : mote.getLinks()) {
                if (selectedPower_DeltaIoTv1.stream()
                    .anyMatch(l -> l.first == link.getSource() && l.second == link.getDestination())) {
                    features.put((int) link.getPower());
                }
            }
        }
        
        // Add the distribution values for certain links (links from motes with 2 parents)
        for (Mote mote : option.system.motes.values()) {
            for (Link link : mote.getLinks()) {
                if (selectedDist_DeltaIoTv1.stream()
                    .anyMatch(l -> l.first == link.getSource() && l.second == link.getDestination())) {
                    features.put((int) link.getDistribution());
                }
            }
        }
        
        // Add the load for motes 10 and 12
        for (TrafficProbability traffic : env.motesLoad) {
            if (selectedLoad_DeltaIoTv1.stream().anyMatch(o -> o == traffic.moteId)) {
                features.put((int) traffic.load);
            }
        }
        
        return features;
    }

    public JSONArray selectFeaturesDeltaIoTv2(AdaptationOption option, Environment env) {
        throw new UnsupportedOperationException("Feature selection currently unavailable for DeltaIoTv2.");
    }
}
    
    