/*
* Predicts adaption options which fullfill goal(s).
* This is called in the end of the analysis function
* It connects the mapek loop to the model checker and the learner
* Watch out, the goals are also hardcoded in the planning step.
*/


package smc;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.google.gson.Gson;

import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.BasicResponseHandler;
import org.apache.http.impl.client.HttpClientBuilder;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.io.OutputStreamWriter;

import mapek.AdaptationOption;
import mapek.Environment;
import mapek.Link;
import mapek.Mote;
import mapek.SNR;
import mapek.TrafficProbability;
import util.ConfigLoader;
import java.io.IOException;
import java.util.Properties;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import smc.SMCChecker;


// Ik denk dat hij na iedere cycle print hoeveel adaption options er 
// werden geselecteerd als die die voorspelt zijn om aan de goal te voldoen


//TODO: make a signal to signal the end of the cycles so the ML can delete its models
// or make an initialise or something so he deletes it in the first step
// Ik denk dat Federico dit al deed

// TODO: make it able to abort the whole proces when you receive an error from the ML
// It will make debugging much easier.
//Als deze class wordt gebruikt in de feedbackloop wordt er eerst
// setAdaptationOptions op opgeroepen
// Daarna wordt er startVerification op opgeroepen
// Ik denk dat dat het startsein heeft om alles te doen.

public class SMCConnector {


	List<AdaptationOption> adaptationOptions;


	// De ruis en packets op een netwerk op een moment
	Environment environment;

	//Modelchecker denk ik
	SMCChecker smcChecker = new SMCChecker();

	List<Goal> goals;

	List<AdaptationOption> verifiedOptions;

	// aantal training sycles
	final int TRAINING_CYCLE = ConfigLoader.getInstance().getAmountOfLearningCycles();
	int cycles = 1;
	Mode mode;
	TaskType taskType;

	// for collecting raw data with the activform mode
	JSONObject rawData;
	JSONArray features;
	JSONArray targets;

	//Gewoon de modes van hem op een string zetten
	public enum Mode {
		TRAINING("training"), 
		TESTING("testing"), 
		ACTIVFORM("activform"), 
		COMPARISON("comparison"),
		// The new mladjustment mode is similar to the comparison mode.
		// The difference between the two is that mladjustment also checks for
		// the adjustments made to the learners after online learning every cycle.
		// This data is then send to the python server to be saved in an output file.
		MLADJUSTMENT("mladjustment");

		String val;

		Mode(String val) {
			this.val = val;
		}

		public static Mode getMode(String value) {
			for (Mode mode : Mode.values()) {
				if (mode.val.equals(value)) {
					return mode;
				}
			}
			throw new RuntimeException(String.format("Run mode %s is not supported.", value));
		}
	}




	// Het type machine learning 
	public enum TaskType {
		CLASSIFICATION("classification"), 
		REGRESSION("regression"), 

		// I will use this for multiclass classification
		MULTICLASS("multiclass"),
		
		// None is used in case no learning task needs to be performed
		// by the learners at the server side (e.g. when just saving data).
		NONE("none");
		
		String val;

		TaskType(String val) {
			this.val = val;
		}

		public static TaskType getTaskType(String value) {
			for (TaskType taskType : TaskType.values()) {
				if (taskType.val.equals(value)) {
					return taskType;
				}
			}
			throw new RuntimeException(String.format("Task type %s is not supported.", value));
		}
	}


	//TODO: dit stond hier nog niet, waarom is dit nu wel nodig?
	public SMCConnector() {
		// Load the configurations specified in the properties file (mode and tasktype)
		ConfigLoader configLoader = ConfigLoader.getInstance();
		mode = configLoader.getRunMode();
		taskType = configLoader.getTaskType();
		goals = initGoals(SMCChecker.DEFAULT_CONFIG_FILE_PATH);
		rawData = new JSONObject();
		features =  new JSONArray();
		targets = new JSONArray();
		rawData.put("features", features);
		rawData.put("targets", targets);

	}


	public void setAdaptationOptions(List<AdaptationOption> adaptationOptions, Environment environment) {
		this.adaptationOptions = adaptationOptions;
		this.environment = environment;
	}

	//TODO: graag train of thought hier
	public void startVerification() {
		switch (mode) {
			case ACTIVFORM:
				activform();
				break;
			case COMPARISON:
				if (cycles <= TRAINING_CYCLE) {
					int space = 0;
					for (AdaptationOption adaptationOption : adaptationOptions) {
						smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
								adaptationOption.verificationResults);
						if (adaptationOption.verificationResults.packetLoss < 10.0) {
							space++;
						}
					}
					System.out.print(";" + space);
					send(adaptationOptions, TaskType.CLASSIFICATION, Mode.TRAINING);
					send(adaptationOptions, TaskType.REGRESSION, Mode.TRAINING);

				} else
					comparison();
				break;
			case MLADJUSTMENT:
				machineLearningAdjustmentInspection(cycles <= TRAINING_CYCLE);
				break;
			default:
				if (cycles <= TRAINING_CYCLE)
					training(taskType);
				else
					testing(taskType);
			}
			cycles++;
	}

	/**
	 * Helper function which adds the predictions of both learning models to their respective JSON arrays.
	 * The predictions are made over the whole adaptation space.
	 * @param classArray The JSON array which will hold the classification predictions.
	 * @param regrArray The JSON array which will hold the regression predictions.
	 */
	private void addPredictionsToJSONArrays(JSONArray classArray, JSONArray regrArray) {
		JSONObject classificationResponse = send(adaptationOptions, TaskType.CLASSIFICATION, Mode.TESTING);
		JSONObject regressionResponse = send(adaptationOptions, TaskType.REGRESSION, Mode.TESTING);

		for (Object item : classificationResponse.getJSONArray("predictions")) {
			classArray.put(Integer.parseInt(item.toString()));
		}
		for (Object item : regressionResponse.getJSONArray("predictions")) {
			regrArray.put(Float.parseFloat(item.toString()));
		}
	}

	/**
	 * FIXME remove later on, here for data analysis
	 *
	 * Similar execution as comparison, but also tracks the adjustments
	 * made to the regression/classification predictions after online learning.
	 * @param training Specifies whether the current cycle is still a training cycle.
	 */
	private void machineLearningAdjustmentInspection(boolean training) {
		JSONObject adjInspection = new JSONObject();
		adjInspection.put("packetLoss", new JSONArray());
		adjInspection.put("energyConsumption", new JSONArray());
		adjInspection.put("regressionBefore", new JSONArray());
		adjInspection.put("classificationBefore", new JSONArray());
		adjInspection.put("regressionAfter", new JSONArray());
		adjInspection.put("classificationAfter", new JSONArray());

		if (cycles == 1) {
			// At the first cycle, no regression or classification output can be retrieved yet
			// -> use -1 as dummy prediction values
			IntStream.range(0, adaptationOptions.size()).forEach(i -> {
				adjInspection.getJSONArray("regressionBefore").put(-1);
				adjInspection.getJSONArray("classificationBefore").put(-1);
			});
		} else {
			// If not at the first cycle, retrieve the results predicted before online learning
			addPredictionsToJSONArrays(adjInspection.getJSONArray("classificationBefore"),
				adjInspection.getJSONArray("regressionBefore"));
		}


		// Check all the adaptation options with activFORMS
		for (AdaptationOption adaptationOption : adaptationOptions) {
			smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
				adaptationOption.verificationResults);
			adjInspection.getJSONArray("packetLoss").put(adaptationOption.verificationResults.packetLoss);
			adjInspection.getJSONArray("energyConsumption").put(adaptationOption.verificationResults.energyConsumption);
		}


		if (training) {
			// If we are training, send the entire adaptation space to the learners and check what they have learned
			send(adaptationOptions, TaskType.CLASSIFICATION, Mode.TRAINING);
			send(adaptationOptions, TaskType.REGRESSION, Mode.TRAINING);

			addPredictionsToJSONArrays(adjInspection.getJSONArray("classificationAfter"),
				adjInspection.getJSONArray("regressionAfter"));
		} else {
			// If we are testing, send the adjustments to the learning models and check their predictions again
			List<AdaptationOption> classificationTrainOptions = new ArrayList<>();
			List<AdaptationOption> regressionTrainOptions = new ArrayList<>();

			// Parse the classification and regression results from the JSON responses.
			final List<Integer> classificationResults = adjInspection.getJSONArray("classificationBefore")
				.toList().stream()
				.map(o -> Integer.parseInt(o.toString()))
				.collect(Collectors.toList());
			final List<Float> regressionResults = adjInspection.getJSONArray("regressionBefore")
				.toList().stream()
				.map(o -> Float.parseFloat(o.toString()))
				.collect(Collectors.toList());

			// Determine which adaptation options have to be sent back for the specific learners
			for (int i = 0; i < adaptationOptions.size(); i++) {
				if (classificationResults.get(i).equals(1)) {
					classificationTrainOptions.add(adaptationOptions.get(i));
				}
				if (regressionResults.get(i) < 10) {
					regressionTrainOptions.add(adaptationOptions.get(i));
				}
			}

			// In case the adaptation space of a prediction is 0, send all adaptations back for online learning
			if (classificationResults.stream().noneMatch(o -> o == 1)) {
				classificationTrainOptions = adaptationOptions;
			}
			if (regressionResults.stream().noneMatch(o -> o < 10)) {
				regressionTrainOptions = adaptationOptions;
			}

			// Send the adaptation options specific to the learners back for online learning
			send(classificationTrainOptions, TaskType.CLASSIFICATION, Mode.TRAINING);
			send(regressionTrainOptions, TaskType.REGRESSION, Mode.TRAINING);

			// Test the predictions of the learners again after online learning to track their adjustments
			addPredictionsToJSONArrays(adjInspection.getJSONArray("classificationAfter"),
				adjInspection.getJSONArray("regressionAfter"));
		}

		// Send the overall results to be saved on the server
		send(adjInspection, TaskType.NONE, Mode.MLADJUSTMENT);
	}



	//Deze functie wordt opgeroepen in de functie hierboven 
	// als de mode comparisson is en 
	// je begint testen en niet trainen
	void comparison() {

		// zend de adaption naar de machinelearner en ontvang een json terug(met de geselecteerde adaption options?)
		JSONObject classificationResponse = send(adaptationOptions, TaskType.CLASSIFICATION, Mode.TESTING);

		// Hetzelfde voor regressie
		JSONObject regressionResponse = send(adaptationOptions, TaskType.REGRESSION, Mode.TESTING);
		
		// Uit het jsonobject die je terugkrijgt van classification haal je de adaptation space (features/option? of nog iets?) 
		// en zet ze om in een string om ze dan in een int te zetten.
		// Ik denk dat dit het aantal is van de geselecteerde adaptation options door machine learners
		int classificationAdaptationSpace = Integer.parseInt(classificationResponse.get("adaptation_space").toString());
		
		// Hetzelfde voor regressie
		int regressionAdaptationSpace = Integer.parseInt(regressionResponse.get("adaptation_space").toString());
		
		//Print de 2 groottes
		System.out.print(";" + classificationAdaptationSpace + ";" + regressionAdaptationSpace);


		// De predictions
		//De eerste is de prediction van in welke class (0 of 1, de ene is succes goal en de andere niet) de configuraties zitten.
		ArrayList<Integer> classificationPredictions = new ArrayList<>();
		
		//Hetzelfde voor regressie
		ArrayList<Float> regressionPredictions = new ArrayList<>();
		
		//Dit zijn de adaptation opties die corresponderen met de 2 arrays hierboven 
		List<AdaptationOption> classificationTrainingOptions = new LinkedList<>();

		List<AdaptationOption> regressionTrainingOptions = new LinkedList<>();

		// Je maakt/leest uit hier een "jsonarray?" van de classepredicties van de classification learner
		// om ze dan om te zetten in een javalijst
		JSONArray arr = classificationResponse.getJSONArray("predictions");
		
		for (int i = 0; i < arr.length(); i++) {
			
			// zet de JSONArray om in een java lijst
			classificationPredictions.add(Integer.parseInt(arr.get(i).toString()));
		}

		//Hier hetzelfde, je leest de prediction uit in een dummy array en zet ze dan in de lijst waar je mee gaat werken.
		arr = regressionResponse.getJSONArray("predictions");
		for (int i = 0; i < arr.length(); i++) {
			regressionPredictions.add(Float.parseFloat(arr.get(i).toString()));
		}

		// Je initialiseerd de grootte van de space die je zal doorzenden naar de modelchecker 
		// Bij nader inzien lijkt dit het aantal adaptations te zijn die voldoen aan de 
		// Goal van sarpreet
		// Je gebruikt de voorspellingen van activforms als waarheid om tegen te controleren
		// Dus als ze deze als correcte voorspelling beschouwen
		// Is de het aantal adaptions die aan de goal voldoen
		int activformAdapationSpace = 0;

		// een index die je later gebruikt
		int index = 0;

		// Init een jsonobject met een key voor de classifcation classevoorspellingen
		// een key voor de regressie voorspellingen en een verse key met een vers array
		// voor de packetloss en energyconsumption voor later
		JSONObject comparison = new JSONObject();
		comparison.put("packetLoss", new JSONArray());
		comparison.put("energyConsumption", new JSONArray());
		comparison.put("classification", classificationResponse.getJSONArray("predictions"));
		comparison.put("regression", regressionResponse.getJSONArray("predictions"));


		// Overloop alle adaptions
		//TODO: goals aanpassen
		for (AdaptationOption adaptationOption : adaptationOptions) {
			
			// Het ziet ernaar uit dat dit alles (class, regr and alles) doorgeeft naar de
			// checker en deze de effectieve resultaten van de simulator doorgeeft naar de 
			// verification results van die adaptation options
			//Of tock niet... het ziet ernaar uit dat die gewoon de voorspelling zijn van activforms
			// Of toch wel... ik weet het niet
			// Ik weet wel dat ze het verder gebruiken als echte oplossing om te vergelijken
			smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
					adaptationOption.verificationResults);
			
			// als de goal voldaan is (1) voeg toe aan adaption options
			// Ik weet nog niet wat die adaptionspace net is dus over het 2e deel kan ik niks zeggen
			if (classificationPredictions.get(index) == 1 || classificationAdaptationSpace == 0) {
				classificationTrainingOptions.add(adaptationOption);
			}

			// Zelfde voor regression
			if (regressionPredictions.get(index) < 10.0 || regressionAdaptationSpace == 0) {
				regressionTrainingOptions.add(adaptationOption);
			}

			// Het aantal adaption options waarvan de modelchecker zegt
			// Dat het beter dan 10 zal zijn. Denk ik ...
			if (adaptationOption.verificationResults.packetLoss < 10.0) {
				activformAdapationSpace++;
			}


			// Je voegt de voorspelling/echte waarden van de goal packetloss en engergyconsumption toe toe aan de comparison object van hiervoor
			// Het ziet ernaar uit dat er zelfs geen ding is voor latency
			// Alhoewel er 5 QoS metrieken zijn vvan de simulator
			comparison.getJSONArray("packetLoss").put(adaptationOption.verificationResults.packetLoss);
			comparison.getJSONArray("energyConsumption").put(adaptationOption.verificationResults.energyConsumption);
			
			//Je verhoogt de index om naar de volgende adaption voorspellingen te kijken
			index++;
		}


		System.out.print(";" + activformAdapationSpace);
		
		//  Hier gaan ze weer van alles verzenden naar de learner, zie send voor je dit uitvogeld
		send(classificationTrainingOptions, TaskType.CLASSIFICATION, Mode.TRAINING);

		//TODO:PAS OP waarom zend dit altijd training? het wordt opgeroepen in de testfase?????
		//Ik denk dat het is omdat je de oplossingen terugzend om er online van te leren
		//Maar kan ook zijn van niet omdate je die pars dinges doorzend en dat zijn niet de oplossing denk ik
		send(regressionTrainingOptions, TaskType.REGRESSION, Mode.TRAINING);

		//Save the outcome
		send(comparison, TaskType.REGRESSION, Mode.COMPARISON);
	}

	// Dit is natuurlijk de fucntei om een training cycle te doen
	void training(TaskType taskType) {
		// int space = 0;

		//Je bent aan het trainen en maakt nog geen prediction
		// Dus moet de model checker eerst nog al het werk doen.
		for (AdaptationOption adaptationOption : adaptationOptions) {
			smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
					adaptationOption.verificationResults);
			// if (adaptationOption.verificationResults.packetLoss < 10.0) {
			// space++;
			// }
		}
		// System.out.print(";" + space);

		//Je zend de data en type door om te training
		// adaptionsoptions is a list, so this gets parsed.
		send(adaptationOptions, taskType, Mode.TRAINING);
	}

	/*
	*
	* Dit is natuurlij de functie om een test cycle te doen
	*
	*/
	void testing(TaskType taskType) {

		// Je zend de adaptationOptions door naar de machine learner met het soort leren en testen
		// en krijgt het antwoord terug in een json
		JSONObject response = send(adaptationOptions, taskType, Mode.TESTING);
		
		//Grootte van?? Mischien zend deze enkel de geselecteerde adaptions terug en meet je zo hoeveel er zijn nu
		int adaptationSpace = Integer.parseInt(response.get("adaptation_space").toString());
		System.out.print(";" + adaptationSpace);

		// Initialising array for prediction and estimates of the qos/goals
		ArrayList<Float> predictions = new ArrayList<>();
		List<AdaptationOption> qosEstimates = new LinkedList<>();

		// for counting 8 classes in multiclass
		int[] mclass = {0, 0, 0, 0, 0, 0, 0, 0};

		// Blijkbaar krijg je dan een json terug met een prediction key die een lijst teruggeeft
		// Hieronder zet je die lijst om naar een java lijst
		JSONArray arr = response.getJSONArray("predictions");
		for (int i = 0; i < arr.length(); i++) {
			if(taskType == TaskType.MULTICLASS)
			{
				mclass[Integer.parseInt(arr.get(i).toString())]++;
			}
			predictions.add(Float.parseFloat(arr.get(i).toString()));
		}

		int nbCorrect = 0;
		// Here I set nbCorrect to the highest ammount of 
		// goals predicted correct for every option in the adaption space
		if (taskType == TaskType.MULTICLASS)
		{
			// are there adaptions which fullfill all goals?
			if(mclass[7] > 0) nbCorrect = 3;
			// are there adaptions with 2 goals fullfilled?
			else if(mclass[6] + mclass[5] + mclass[4] > 0) nbCorrect = 2;
			// 1 correct
			else if(mclass[3] + mclass[2] + mclass[1] > 0) nbCorrect = 1;
			// 0
			else nbCorrect = 0;
		}

		// Zet de incrmentloper op 0 als init
		int i = 0;

		//Overloop de adaption opptions
		for (AdaptationOption adaptationOption : adaptationOptions) {
			
			// Als de machine learner adaptation options voorspelt die aan de
			// goals voldoen, doe:
			// if nbcorrect == 0 , this will also be 0.
			if (adaptationSpace != 0) {

				boolean isPredictedCorrect = false;

				//TODO: als je hier een andere classtype toevoegd zal dit niet meer werken.
				//Hou dat in het achterhoofd
				// Dat in de if geeft True of False terug door die predict.get
				// En dat .. ? .. :.. zorgt gewoon dat het
				// voor classificatie 1.0 voorspeld of minder dan 10 voor regressie
				//if (taskType == TaskType.CLASSIFICATION ? predictions.get(i) == 1.0 : predictions.get(i) < 10.0) {
				//	
				//	//Indien voorspelt dat hij aan de goals voldoet zal de modelchecker hem draaien
				//	smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
				//			adaptationOption.verificationResults);

					// options terug toevoegen om door te zenden naar de learner om te online learnen
					// Hij learned dus de predicties van de SMC en niet wat er effectief gebeurde
				//	qosEstimates.add(adaptationOption);
				//}

				if (taskType == TaskType.CLASSIFICATION)
				{
					if(predictions.get(i) == 1.0) isPredictedCorrect = true;
				}
				else if(taskType == TaskType.REGRESSION)
				{
					// I will work with another regression tasktype
					if(predictions.get(i) < 10) isPredictedCorrect = true;
				}
				else if(taskType == TaskType.MULTICLASS)
				{
					double pred = predictions.get(i);

					if( nbCorrect == 3)
					{
						if(pred == 7.0) isPredictedCorrect = true;
					}
					else if( nbCorrect == 2)
					{
						if(pred == 6.0 || pred == 5.0 || pred == 4.0) isPredictedCorrect = true;
					}
					else
					{
						if(pred == 3.0 || pred == 2.0 || pred == 1.0) isPredictedCorrect = true;
					}

				}

				if(isPredictedCorrect)
				{
					//Indien voorspelt dat hij aan de goals voldoet zal de modelchecker hem draaien
					smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
							adaptationOption.verificationResults);

					// options terug toevoegen om door te zenden naar de learner om te online learnen
					// Hij learned dus de predicties van de SMC en niet wat er effectief gebeurde
					qosEstimates.add(adaptationOption);

				}
				// Als het voorspeld werd dat er niet aan de goals voldaan is
				else {

					// Ik denk dat dit een manier is om er voor te zorgen dat de model checker dit zeker niet selecteerd
					adaptationOption.verificationResults.packetLoss = 100.0;
				}


			// Dit doe je als de adaptionspace lees is.
			// Dit is dus voor als de machine learner voorspelt dat er geen enkele
			// Zal voldoen aan de goal(s) 
			// je laat de model checker dan gewoon alles checken om de beste eruit te halen.
			} else {
				smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
						adaptationOption.verificationResults);
				qosEstimates.add(adaptationOption);
			}
			i++;
		}

		//Ik denk dat je hier de oplossingen/model checker voorspelling die je waar acht, terug doorzend
		// naar de machine learner om ervan te leren als online learning.
		// Daarom Training.
		send(qosEstimates, taskType, Mode.TRAINING);
	}

	//Deze functie voert gewoon actiforms uit door alles terug te zenden.
	//
	void activform() {
		try 
		{
			File dat;

			//System.out.println("Before getting paths");
			String datPath = Paths.get(System.getProperty("user.dir"), "activforms", "log", "rawData.txt").toString();
			//String jdatPath = Paths.get(System.getProperty("user.dir"), "activforms", "log", "rawData.json").toString();
			
			
			if (cycles == 1)
			{
				dat = new File(datPath);

				//System.out.println("Before file deletion");
				if(dat.isFile())
				{
					dat.delete();
				}
				dat.createNewFile();
			}
			else
			{
				dat = new File(datPath);
			}

			//System.out.println("init printwriter");
			PrintWriter writer = new PrintWriter(new OutputStreamWriter( new FileOutputStream(dat, true), StandardCharsets.UTF_8));
			System.out.print(";" + adaptationOptions.size());



			JSONArray dummyFeatures;
			JSONObject qos;

			Mote mote;

			// prints the adaption options for the link.
			//System.out.println("writing to txt");
			writer.println(environment);

			//System.out.println("before iterating");
			for (AdaptationOption adaptationOption : adaptationOptions) {
				smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
						adaptationOption.verificationResults);

				writer.println(adaptationOption);

				// get features
				// iterate over all 15 motes (is hashmap and want to be sure of order)
				// TODO:HARDCODED
				// TODO: for some reason there is no 1 mote but 2-15
				dummyFeatures = new JSONArray();
				//System.out.println("before motes");
				//for(int s : adaptationOption.system.motes.keySet()) System.out.println(s);
				for (int i = 2; i <=15; i++)
				{
					mote = adaptationOption.system.getMote(i);
					//System.out.println("at mote "+i);
					for (TrafficProbability t : environment.motesLoad)
					{
						//System.out.println("at traffic");
						//System.out.println("traffic "+t.moteId);
						//System.out.println("mote"+mote.getMoteId());
						if(t.moteId == mote.getMoteId())
						{
							//System.out.println("at loading traffic");
							dummyFeatures.put(t.load);
						}
					}
					//System.out.println("before links");
					for (Link link : mote.getLinks())
					{
						// power is usefull if we learn it or not, 
						// because it does change every cycle.
						// Sarpreet left this out, which is not so smart
						// because it will confuse the learner
						// because the same snr will sometimes have different
						// class without anything else changing.
						dummyFeatures.put(link.getPower());
						dummyFeatures.put(link.getDistribution());
						dummyFeatures.put(environment.getSNR(link));
					}
				}
				//System.out.println("before putteing features");
				features.put(dummyFeatures);


				//System.out.println("before qos");
				
				// get qos
				// This has to be processed for every mode, thats why I add objects.
				qos = new JSONObject();
				qos.put("packetLoss", adaptationOption.verificationResults.packetLoss);
				qos.put("latency", adaptationOption.verificationResults.latency);
				qos.put("energyConsumption", adaptationOption.verificationResults.energyConsumption);
				targets.put(qos);

				

				
			}
			writer.close();
			
			//System.out.print("before saving json");

			// write at the end of all the cycles
			if (this.cycles == ConfigLoader.getInstance().getAmountOfCycles())
			{
				FileWriter jsonWriter = new FileWriter(Paths.get(System.getProperty("user.dir"), "activforms", "log", "rawData.json").toString());
				jsonWriter.write(rawData.toString());
				jsonWriter.flush();
				jsonWriter.close();
			}
		}
		catch (Exception e) 
		{
			System.out.println("Problem writing to file.\n");
		}
	}

	// Ik denk dat deze functie de data met features en targets omzet en klaarmaakt om 
	// door te zenden naar de learner
	JSONObject parse(List<AdaptationOption> adaptationOptions, TaskType taskType) {
		
		//Json object dat je dan kan gebruiken om de data in te bundelen
		JSONObject dataset = new JSONObject();

		//De arrays die het vanzelfsprekende voorstellen
		JSONArray features = new JSONArray();
		JSONArray target = new JSONArray();

		// Je voegt de fearues toe aan de datasaet json
		dataset.put("features", features);
		dataset.put("target", target);

		// Voor iedere adaption option
		for (AdaptationOption adaptationOption : adaptationOptions) {
			
			// Iedere item krijgt een array
			JSONArray item = new JSONArray();

			// Hier voeg je de gemeten target toe

			// Je voegt de geobserveerde/ door activforms voorspelde targets toe van de cycle
			// voor regressie voer je de letterlijke waarde in, voor classification de corresponderende klasse
			// TODO: pas dit aan naar jou klassen
			if (taskType == TaskType.CLASSIFICATION) {
				target.put(adaptationOption.verificationResults.packetLoss < 10.0 ? 1 : 0);
			} 
			// makes corresponding classes for multiclass verification
			// There are 8 possible combinations for the goals
			// The succes of a goal represent one bit in 
			// 3 bits (a power of 2).
			// So all possible combination can be represented in 3 bits.
			// So a  number from 0-7 = class.
			else if(taskType == TaskType.MULTICLASS)
			{
				int APClass = 0;

				//What I do here is dirty
				// but welcome to this project
				Goal pl = goals.get(1), en = goals.get(1), la = goals.get(1);
				for(Goal g : goals)
				{
					if(g.getTarget() == "packetLoss") pl = g;
					else if(g.getTarget() == "latency") la = g;
					else en = g;
				}

				if( pl.evaluate(adaptationOption.verificationResults.packetLoss)) APClass += 1;
				if( en.evaluate(adaptationOption.verificationResults.energyConsumption)) APClass += 2;
				if( la.evaluate(adaptationOption.verificationResults.latency)) APClass += 4;

				target.put(APClass);
			}
			else {
				target.put((int) adaptationOption.verificationResults.packetLoss);
			}

			
			// Vanaf hier voeg je de features toe

			//TODO: vraag wat dit allemaal betekent.

			// Voeg de snr toe van de links (van die adaption???)
			for (SNR snr : environment.linksSNR) {
				item.put((int) snr.SNR);
			}

			// TODO: veralgemeen voor note met ouders

			// Distributiefactor toevoegen 
			// Hij doet dit enkel voor de motes 7, 10 en 12 omdat die in zijn kleine modelletje enkel degene zijn met 2 parent links
			for (Mote mote : adaptationOption.system.motes.values()) {
				for (Link link : mote.getLinks()) {
					if (link.getSource() == 7 || link.getSource() == 10 || link.getSource() == 12) {
						item.put((int) link.getDistribution());
					}
				}
			}



			//Voeg de load toe
			//TODO: waarom enkel aan 12 en 10?
			// waarschijnlijk omdat dit distributie is in plaats van load.
			// Lekker gehardcode
			for (TrafficProbability traffic : environment.motesLoad) {
				if (traffic.moteId == 10 || traffic.moteId == 12) {
					item.put((int) traffic.load);
				}
			}

			// Je voegt de features samen toe
			features.put(item);
		}

		// Returns the dataset with the features and targets
		return dataset;
	}


	//TODO: waarvoor worden deze 2 gebruikt?
	private JSONObject send(List<AdaptationOption> adaptationOptions, TaskType taskType, Mode mode) {
		return send(parse(adaptationOptions, taskType), taskType, mode);
	}

	private JSONObject send(JSONObject dataset, TaskType taskType, Mode mode) {
		return send(dataset, taskType.val, mode.val);
	}

	// Hier zend je de json door naar de learner, vaak zend je de features en targets door zoals ze 
	// gemaakt zijn hierboven in parse
	// Echter weet ik dat je ook ergens die comparisson doorzend 
	// Die een andere vorm van json doorzend
	JSONObject send(JSONObject dataset, String taskType, String mode) {
		
		try {
			
			// Je maakt een http client aan
			HttpClient client = HttpClientBuilder.create().build();
			
			// Met http post methode zend je iets door naar de andere kant
			// Hier bereid je dat voor
			// De andere kant is hier dus port 8000 op de localhost
			// Je kan parameters meegeven in de url na dat ? dan 
			// Denk ik dat je begint met met de naam van je eerste
			// variabele, dan een = en dan tussen "" de waarde die je toewijst
			// Dan waarschijnlijk zet je een & en begin je hetzelfde opnieuw
			HttpPost http = new HttpPost("http://localhost:8000/?task-type=" + taskType + "&mode=" + mode + "&cycle=" + cycles);
			
			// The entity of a http post is the payload
			// Here you give as payload the json dataset to send in the form of a string
			http.setEntity(new StringEntity(dataset.toString()));


			// The header holds some info about the data
			// So here you say the content-type of the payload = entity
			// Is a json file
			http.setHeader("Content-Type", "application/json");
			
			// start is the current time
			long start = System.currentTimeMillis();

			//You execute the "http" which is the Post thing you just set up through the client you 
			// defined in the beginning
			// This returns the response of the machine learner
			// I think you get a json in the form of a string back
			// Which turns it back into a json 
			JSONObject response = new JSONObject(client.execute(http, new BasicResponseHandler()));
			
			// Prints the time it took for the machine learner to send an answer back
			System.out.print(";" + (System.currentTimeMillis() - start));

			// Returns the json returned by the machine learner
			return response;

		// If something failed, print the problem and return null
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	// reads the goals from the properties file and returns a list of them as Goal objects
	public static List<Goal> initGoals(String configPath)
	{

		Properties prop = new Properties();
		InputStream input = null;
		List<Goal> rgoals = new LinkedList<>();

		try {

			// TODO adjust this part to use the ConfigLoader class
			File configFile = new File(configPath);
			if (!configFile.exists()) {
				throw new RuntimeException("SMCConfig.properties file not found at following path:" + configPath);
			}


			input = new FileInputStream(configPath);

			// load a properties file
			prop.load(input);

			// get the property value
			// load the requirements, aka the models to be predicted by the smc
			String targets[] = prop.getProperty("targets").split(",");
			String operators[] = prop.getProperty("operators").split(",");
			String thressholds[] = prop.getProperty("thressholds").split(",");

			double numThress[] = new double[thressholds.length];
			for(int i = 0; i < thressholds.length; i++)
			{
				
				numThress[i] = Double.parseDouble(thressholds[i].trim());

			}

			
			for(int i = 0; i < targets.length; i++)
			{

				rgoals.add(new Goal( targets[i].trim(),
				operators[i].trim(), numThress[i]));

			}

			return rgoals;

		}
		catch (IOException ex) {
			ex.printStackTrace();
			return null;
		}
		finally {
			if (input != null) {
				try {
					input.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		
		}

	}
}


