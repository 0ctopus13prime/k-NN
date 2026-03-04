/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.bbq;

import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.AcceptDocs;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.nativeindex.bbq.BBQReader;
import org.opensearch.knn.index.codec.nativeindex.bbq.BBQWriter;
import org.opensearch.knn.index.codec.nativeindex.bbq.BinarizedByteVectorValues;
import org.opensearch.knn.index.codec.nativeindex.bbq.Lucene102BinaryFlatVectorsScorer;
import org.opensearch.knn.index.codec.nativeindex.bbq.OffHeapBinarizedVectorValues;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.jni.FaissService;
import org.opensearch.knn.memoryoptsearch.faiss.FaissMemoryOptimizedSearcher;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Program 2: Faiss BBQDistanceComputer validation.
 * Ingests vectors through BBQWriter -> BBQReader -> FaissBBQFlat via JNI,
 * then calls bbqValidationScan to do symmetric scoring on the C++ side.
 */
public class FaissBBQRecallValidationTests extends KNNTestCase {

    // ===== CONFIGURE THESE =====
    private static final String DATA_PATH = "/Users/kdooyong/workspace/io-opt/tmp/vectors.bin";
    private static final int TOP_K = 100;
    private static final int QUERY_VECTOR_ORDINAL = 0;
    // ===========================

    public void testFaissBBQRecall() throws Exception {
        float[][] vectors = VectorDataLoader.loadVectors(DATA_PATH);
        int dimension = vectors[0].length;
        int numVectors = vectors.length;

        System.out.println("Loaded " + numVectors + " vectors with dimension " + dimension);

        Path tmpDir = Files.createTempDirectory("faiss_bbq_validation");
        byte[] segmentId = StringHelper.randomId();
        String segmentName = "test";
        try (Directory directory = new MMapDirectory(tmpDir)) {
            SegmentInfo segmentInfo = new SegmentInfo(
                directory, Version.LATEST, Version.LATEST, segmentName, vectors.length,
                false, false, null, new HashMap<>(), segmentId, new HashMap<>(), null
            );

            final FieldInfo fieldInfo = new FieldInfo(
                "test_field",
                0,
                false,
                false,
                false,
                IndexOptions.NONE,
                org.apache.lucene.index.DocValuesType.NONE,
                DocValuesSkipIndexType.NONE,
                -1,
                new HashMap<>(),
                0,
                0,
                0,
                dimension,
                VectorEncoding.FLOAT32,
                VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
                false,
                false
            );

            // Step 1: Write vectors through BBQWriter to produce .veb files
            writeVectors(directory, vectors, segmentInfo, fieldInfo);

            // Step 2: Read back via BBQReader and ingest into FaissBBQFlat via JNI
            final long indexMemoryAddress = ingestIntoFaiss(directory, dimension, numVectors, segmentInfo, fieldInfo);

            // Step 3: Call JNI validation scan
            FaissService.bbqValidationScan(indexMemoryAddress, TOP_K, QUERY_VECTOR_ORDINAL);

            // Step 4: Write HNSW index to file
            String faissIndexFileName = "faiss-bbq-validation.hnsw";
            try (var indexOutput = directory.createOutput(faissIndexFileName, IOContext.DEFAULT)) {
                IndexOutputWithBuffer outputWithBuffer = new IndexOutputWithBuffer(indexOutput);
                Map<String, Object> writeParams = new HashMap<>();
                writeParams.put("name", "hnsw");
                writeParams.put("data_type", "float");
                writeParams.put("index_description", "BHNSW16,Flat");
                writeParams.put("spaceType", "innerproduct");
                Map<String, Object> subParams = new HashMap<>();
                writeParams.put("parameters", subParams);
                subParams.put("ef_search", 256);
                subParams.put("ef_construction", 256);
                subParams.put("m", 16);
                subParams.put("encoder", Collections.emptyMap());
                subParams.put("indexThreadQty", 1);
                FaissService.writeBBQIndex(outputWithBuffer, indexMemoryAddress, writeParams);
            }

            // Step 5: Load index back and search with FaissMemoryOptimizedSearcher
            FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
            SegmentReadState readState = new SegmentReadState(
                directory, segmentInfo, fieldInfos, IOContext.DEFAULT
            );

            System.out.println("\nANN search results:");
            try (var indexInput = directory.openInput(faissIndexFileName, IOContext.DEFAULT)) {
                try (FaissMemoryOptimizedSearcher searcher = new FaissMemoryOptimizedSearcher(indexInput, fieldInfo, readState)) {
                    final float[] queryVector = vectors[QUERY_VECTOR_ORDINAL];
                    double[] queryVector1 = new double[]{
                        0.24765340983867645, -0.09663575142621994, -0.09608439356088638, -0.23322473466396332, 0.12766943871974945, 0.38648271560668945, 0.7653201222419739, -0.25100255012512207, 0.11393576115369797, 0.13293230533599854, -0.5390275120735168, -0.33049023151397705, -0.4746643006801605, -0.10072320699691772, -0.6021131277084351, -0.16435784101486206, 0.45183518528938293, -0.0385432131588459, -0.09143006056547165, 0.0554346963763237, -0.4146067202091217, 0.3839111328125, 0.30511295795440674, 0.04690251871943474, -0.07808893173933029, -0.24753965437412262, 0.09537383913993835, 0.24583248794078827, 0.19480770826339722, 0.2995428144931793, 0.7694968581199646, 0.08617742359638214, 0.08282379806041718, 0.3739044666290283, -0.5176703929901123, 0.04634667560458183, 0.04628339782357216, 0.2276340126991272, -0.2408454716205597,0.7612088322639465, 0.18359249830245972, 0.16596023738384247, 0.6570728421211243, 0.11740405112504959, 0.6211836338043213, -0.42610064148902893, 0.10026377439498901, 0.002928501693531871, -0.07425213605165482, 0.15369360148906708, 0.1712190806865692, -0.8445386290550232, -0.43232032656669617, 0.18752071261405945, -0.6306326389312744, 0.7796579003334045, -0.2831922173500061, 0.48439815640449524, -0.24311557412147522, -0.01737569458782673, 0.014272511005401611, 0.20538318157196045, -0.04206249862909317, 0.21355769038200378, 0.32843130826950073, -0.14527255296707153, 0.00518816476687789, 0.5612276792526245, 0.300973504781723, -0.024140797555446625, -0.4192330539226532, 0.608636736869812, 0.6987252831459045, 0.18647132813930511, -0.08210517466068268, -0.32546061277389526, -0.38606834411621094, 0.1642586588859558, 0.07549094408750534, -0.4296349883079529, 0.12705214321613312, 0.4154076874256134, 0.13750633597373962, 0.15292072296142578, 0.36404919624328613, 0.4959639310836792, 0.13429789245128632, -0.31279346346855164, -0.3133848309516907, 0.7046182751655579, -0.49892041087150574, 0.1338067650794983, -0.14128300547599792, -0.17951563000679016, 0.12840360403060913, 0.33799389004707336, 0.08421096950769424, -0.026834936812520027, 0.2361137568950653, -0.5780436396598816, 0.4951629638671875, -0.18847504258155823, -0.3049788773059845, 0.39235222339630127, 0.13496465981006622, -0.38057389855384827, -0.29219380021095276, -0.32404178380966187, 0.13254688680171967, 0.26503321528434753, 0.6418581604957581, -0.024853115901350975, 0.04453379660844803, -0.15341436862945557, 0.1881672739982605, -0.004332063253968954, 0.32461488246917725, -0.38507887721061707, -0.2928857207298279, -1.5226598978042603, 0.29300498962402344, 0.209535151720047, -0.36450791358947754, 0.5050246715545654, 0.25192439556121826, 0.47665131092071533, 0.3955407440662384, 0.06367120891809464, 0.5672878623008728, 0.5818274617195129, 0.3862561285495758, 0.37630894780158997, -0.14366686344146729, 0.8082849979400635, 0.6054475903511047, 0.028180420398712158, 0.06247466802597046, -0.28592875599861145, 0.30655351281166077, -0.15462811291217804, 0.2699427902698517, -0.5410604476928711, 0.09710974246263504, 0.36587774753570557, -0.30923399329185486, 0.5458383560180664, 0.38733118772506714, 0.13318398594856262, 0.09207674860954285, -0.09081149101257324, 0.6633712649345398, 0.011056470684707165, -0.45141473412513733, 0.5692074298858643, -0.08901289105415344, 0.01547209918498993, 0.0985550731420517, 0.09167765080928802, -0.2609691917896271, 0.301023006439209, 0.7493418455123901, 0.36366069316864014, 0.5472468733787537, -0.6162874698638916, -0.14394667744636536, 0.028336388990283012, 0.023870011791586876, 0.23933209478855133, 0.4226219654083252, 0.23841199278831482, -0.17456181347370148, 0.043152209371328354, 0.8586561679840088, -0.16096988320350647, 0.28119567036628723, 0.5944649577140808, 0.5778698325157166, 0.3762119710445404, 0.30314919352531433, -0.41869446635246277, 0.056582141667604446, 0.16149064898490906, 0.2815655767917633, -0.02788729965686798, 0.5542513728141785, 0.19460240006446838, 0.17768357694149017, -0.08138548582792282, -0.37011751532554626, 0.4631115198135376, -0.19625115394592285, -0.3160199820995331, 0.9503325819969177, -0.6756766438484192, -0.14317627251148224, 0.7894267439842224, 0.03550991043448448, 0.4170985221862793, -0.5943484306335449, -0.018749086186289787, -0.27358168363571167, -0.32517844438552856, -0.32753610610961914, 0.13885395228862762, 0.6053441166877747, -0.6094104647636414, -0.2362447828054428, 0.36252152919769287, -0.16525691747665405, -0.5177765488624573, -0.3236566483974457, 0.015073597431182861, 0.5440025329589844, 0.12765979766845703, 0.04605673626065254, -0.15574978291988373, -0.15122392773628235, 0.1743326485157013, 0.38281741738319397, -0.15853486955165863, -0.07287033647298813, 0.36802545189857483, -0.4811601936817169, -0.12944777309894562, -0.06330328434705734, -0.10140916705131531, 0.23388533294200897, -0.460124671459198, 0.2367124855518341, 0.24423518776893616, -0.17687952518463135, 0.049932509660720825, 0.36468109488487244, 0.22938303649425507, 0.35935431718826294, 0.25094401836395264, 1.032539963722229, -0.5667144060134888, -0.5759044885635376, 0.48103955388069153, 0.5322788953781128, 0.3366885185241699, -0.013009192422032356, 0.3955143988132477, 0.5718031525611877, 0.06657113134860992, -0.08604322373867035, 0.4741746187210083, 0.13853080570697784, -0.08487770706415176, 0.051899563521146774, -0.18649597465991974, 0.18981121480464935, 0.43128281831741333, 0.044453032314777374, 0.5029452443122864, 0.0871676504611969, -0.2331966608762741, 0.0999302938580513, 0.22114695608615875, 0.8215087652206421, 0.5777884721755981, 0.27441832423210144, 0.14899317920207977, -0.232118621468544, -0.5458422303199768, 0.15616318583488464, 0.40980541706085205, 0.3081861138343811, 0.6943020224571228, 0.11392027884721756, -0.4190567135810852, -0.5680357813835144, 0.47875428199768066, -0.11407122761011124, -0.17391301691532135, 0.09016528725624084, 0.1578240990638733, -0.6311331987380981, 0.3350595235824585, 0.08434807509183884, 0.03633228689432144, 0.039533041417598724, 0.26689043641090393, 0.2854289412498474, 0.4038984775543213, 0.37669476866722107, -0.14469929039478302, -0.5134005546569824, 0.32544130086898804, 0.44363000988960266, 0.48534122109413147, -0.039839256554841995, -0.45365777611732483, -0.06245436519384384, 0.10377835482358932, 0.20971417427062988, 0.027447765693068504, 0.25333139300346375, -0.2997368574142456, 0.49293461441993713, -0.5342702865600586, 0.32291606068611145, -0.22764918208122253, 0.04855861887335777, -0.29606375098228455, -0.5415876507759094, -0.11960268765687943, 0.05850021913647652, 0.5404366254806519, 0.856230616569519, -0.8770688772201538, -0.08030430972576141, 0.6783114671707153, 0.3479757010936737, 0.7943912744522095, 0.3704950213432312, 0.473843514919281, 0.09727932512760162, 0.12401806563138962, -0.19438768923282623, -0.20364739000797272, -0.47310611605644226, -0.07365738600492477, 0.2518293559551239, -0.26653000712394714, -0.3893147110939026, -0.2917870581150055, 1.205811858177185, 0.4355444610118866, -0.22906219959259033, 0.33011361956596375, 0.19945964217185974, -0.02358260005712509, 0.004331895150244236, 0.46212849020957947, 0.7748199701309204, 0.6005537509918213, -0.007689975667744875, -0.276693731546402, -0.04574057087302208, 0.4344390034675598, 0.27283260226249695, 0.21935352683067322, 0.24382895231246948, 0.49139952659606934, 0.169637069106102, -0.22989940643310547, 0.38116082549095154, 0.2952485978603363, 0.26660019159317017, 0.3222828507423401, -0.1905660629272461, 0.023344432935118675, -0.02547089196741581, 0.15006454288959503, 0.32389265298843384, -0.18580122292041779, 0.5059327483177185, -0.26095330715179443, 0.39798861742019653, 0.4110967218875885, 0.15019799768924713, 0.41074973344802856, 0.5038760304450989, 0.11473502963781357, -0.06573130935430527, 0.00021214460139162838, -0.7266342639923096, -0.3206634223461151, 0.13505753874778748, -0.5143436789512634, 0.17659416794776917, 0.01672615297138691, -0.40174129605293274, -0.5119309425354004, -0.33923792839050293, 0.37349921464920044, 0.3152620792388916, -0.34903326630592346, 0.23997355997562408, 0.6341922879219055, 0.15402457118034363, 0.05104971304535866, -0.8220706582069397, -0.009593045338988304, -0.24212008714675903, 0.46616947650909424, 0.5108245611190796, -1.0301804542541504, 0.13605563342571259, 0.04437284544110298, 0.6303166747093201, -0.10881451517343521, 0.006005342584103346, -0.011041399091482162, -0.30013611912727356, 0.1968713402748108, 0.0371912345290184, 0.0362740084528923, -0.2510004937648773, 0.3387879729270935, 0.3538099527359009, 0.10943082720041275, 4.237619876861572, -0.02128603681921959, 0.13217362761497498, 0.19256578385829926, -0.10819773375988007, 0.06181449070572853, 0.833329439163208, 0.07858672738075256, 0.03820367902517319, 0.20419269800186157, 0.10763726383447647, 0.136185422539711, 0.13021892309188843, -0.3154730498790741, 0.19351167976856232, -0.21104229986667633, 0.3801969885826111, -0.03225509077310562, 0.3054850399494171, 0.5813983082771301, -0.2859744429588318, 0.49921226501464844, 0.22660081088542938, -0.07250067591667175, 0.39397862553596497, -0.19242282211780548, -0.3568101227283478, 0.4899333715438843, 0.7247551679611206, 0.17383529245853424, 0.36639463901519775, -0.16908666491508484, 0.2878303825855255, 0.2175680547952652, -0.649475634098053, 0.4205898642539978, -0.08730144798755646, -0.5408536195755005, -0.08340996503829956, 0.20237499475479126, -0.24662268161773682, -0.2693467140197754, -0.2464098185300827, 0.6734289526939392, -0.3252328038215637, -0.38130131363868713, -0.11544033139944077, 0.25156736373901367, 0.0249903853982687, -0.007092765998095274, 0.03048195317387581, -0.04400822892785072, 0.06802280992269516, -0.708257794380188, 0.09163551777601242, 0.6664948463439941, -0.2859621047973633, 0.5375123620033264, 0.013323459774255753, 0.04894453287124634, -0.1586667001247406, 0.17785261571407318, 0.3025832772254944, 0.0036439995747059584, -0.6691962480545044, 0.31716424226760864, 0.10105609148740768, 0.39621326327323914, 0.3807424008846283, 0.10654328018426895, 0.05893085524439812, 0.14514808356761932, 0.5131791830062866, 0.07760630548000336, -0.9240490198135376, 0.11541853845119476, -0.22315889596939087, 0.24233250319957733, 0.45178624987602234, -0.18227143585681915, 0.6038705110549927, -0.3483169674873352, 0.10367067158222198, 0.08434552699327469, -0.5205134749412537, 0.5673834681510925, -0.47715339064598083, -0.4891250431537628, 0.6018624901771545, 0.13457170128822327, 0.05926106125116348, -0.14185942709445953, 0.21094849705696106, 0.2460758239030838, 0.3102455139160156, -0.07803531736135483, 0.14987072348594666, -3.6775896549224854, 0.6639307141304016, 0.20600411295890808, -0.5611270070075989, 0.2160748690366745, -0.3692750334739685, 0.18186123669147491, 0.1232568621635437, -0.8208563327789307, 0.44232693314552307, -0.1679186075925827, -0.27546587586402893, -0.2347102016210556, 0.15735842287540436, -0.10718806087970734, -0.028727930039167404, -0.29217517375946045, 0.49076852202415466, 0.8968214988708496, -0.2934795916080475, 0.2336883395910263,0.22928254306316376, 0.32709795236587524, 0.0016100292559713125, -0.206094890832901, -0.03464542701840401, 0.09166903048753738, -0.1491725593805313, -0.09912530332803726, -0.07007794827222824, -0.3817141056060791, 0.33660992980003357, 0.3085971474647522, -0.2087065726518631, 0.031378988176584244, 0.5493225455284119, 0.19271138310432434, 0.14830780029296875, 0.1095176711678505, 0.127456933259964, 0.15476050972938538, 0.30406004190444946, 0.08237656205892563, 0.3881801962852478, -0.11666485667228699, 0.18847228586673737, -0.37620967626571655, 0.7772200703620911, 0.2710283398628235, 0.12414056062698364, 0.39655596017837524, 0.43144935369491577, -0.3125288486480713, -0.41393905878067017, 0.633978009223938, 0.1500060111284256, -0.09295034408569336, 0.2679935097694397, 0.45122429728507996, 1.0036453008651733, -0.16327962279319763, 0.47285282611846924, 0.5586725473403931, -0.2231447696685791, -0.06630139797925949, 0.19463512301445007, -0.12661075592041016, 0.3380200266838074, 0.8672423958778381, 0.15040592849254608, -0.22306327521800995, 0.03445916250348091, -0.022646933794021606, -0.41062915325164795, 0.39817795157432556, 0.3998207747936249, 0.11166466772556305, 0.2707757353782654, 0.48528844118118286, 0.3289090394973755, 0.05615144595503807, 0.44507238268852234, -0.3793551027774811, 0.3888988494873047, 2.661411762237549, 0.4903482496738434, 2.07334041595459, -0.08469706028699875, 0.40231087803840637, 0.48092684149742126, -0.5107340216636658, -0.021680235862731934, -0.19750061631202698, 0.13457222282886505, 0.5482193231582642, -0.753434419631958, -0.13851535320281982, -0.265987366437912, -0.2842984199523926, -0.05259252339601517, 0.35949528217315674, -1.0404932498931885, -0.5818116664886475, -0.02925979718565941, -0.043239183723926544, -0.544512927532196, -0.09261931478977203, 0.3304114043712616, 0.5048052668571472, 0.12810295820236206, -0.09924086183309555, 0.48246219754219055, 0.1233656033873558, 0.21976613998413086, -0.5042091608047485, -0.6698698997497559, 0.352904736995697, 0.047865983098745346, -0.18604770302772522, 0.36780795454978943, -0.4147203266620636, 4.351893424987793, 0.14860431849956512, 0.12567760050296783, -0.12648287415504456, -0.01485531684011221, 0.09161151945590973, 0.35163289308547974, 0.049358662217855453, 0.023992113769054413, 0.42358726263046265, 0.29597488045692444, 0.766273558139801, 0.23077307641506195, -0.7328016757965088, 0.14501361548900604, -0.5578727126121521, 0.027832048013806343, 0.3194366991519928, 0.259204626083374, 0.04366392269730568, -0.0489138662815094, -0.03914729505777359, 0.31221768260002136, -0.12209348380565643, 0.6559876203536987, 0.10308767110109329, 0.45231592655181885, 0.5117404460906982, -0.021122165024280548, -0.2792167067527771, 0.24962733685970306, 5.095754623413086, 0.3452417254447937, -0.5021525025367737, 0.00849019456654787, 0.007927821017801762, 0.27661824226379395, 0.1433897614479065, 0.41058704257011414, -0.12516751885414124, -0.06980163604021072, -0.008225412108004093, -0.008074709214270115, -0.009512669406831264, 0.6939893364906311, 0.5010160207748413, 0.2464192658662796, -0.5768448710441589, 0.010517618618905544, 0.42363283038139343, -0.2609384059906006, 0.42691177129745483, -0.7562636733055115, 0.10787850618362427, -0.6478713154792786, -0.5431615710258484, -0.45578551292419434, -0.41411325335502625, 0.43459904193878174, -0.35387343168258667, -0.034598417580127716, 0.7359748482704163, 0.5361953973770142, -0.5276833176612854, 0.7088261246681213, 0.04245603829622269, -0.2793833315372467, 0.6806585788726807, -0.46321138739585876, 0.25017091631889343, -0.4070967435836792, 0.19517311453819275, 0.3189278841018677, -0.12623679637908936, -0.011021165177226067, -0.3166584074497223, -0.0398135669529438, 0.11463288217782974, 0.15839187800884247, -0.21857093274593353, -0.2994094789028168, -0.1331561803817749, -0.57526034116745, 0.8509541153907776, -0.06969916820526123, 0.502501904964447, 0.1298467516899109, 0.6333880424499512, -0.21319438517093658, 0.18368792533874512, -0.333601176738739, 0.49538853764533997, 0.037957075983285904, 0.15433132648468018, 0.23146533966064453, 0.490841805934906, -0.006753603462129831, 0.6216419339179993, 0.1435081511735916, 0.8195400238037109, -0.6062000393867493, 0.3366634249687195, 0.09330818057060242, -0.07225733995437622, 0.639017641544342, -0.4213322103023529, -0.22074417769908905, 0.03828934580087662, -0.08789250254631042, 0.1250121146440506, -0.1355300396680832, -0.1326701045036316, -0.3893115222454071, -0.19588850438594818, 0.012783341109752655, 0.24198809266090393, 0.39870119094848633, 0.3720370829105377, -0.14937268197536469, 0.9890362620353699, 0.14451457560062408, 0.13194316625595093, 0.37022262811660767, 0.2778187394142151, 0.2634443938732147, 0.20372122526168823, 0.10275242477655411, -0.0611463338136673, 0.16238880157470703, -0.42774268984794617, 0.009685623459517956, -0.056855231523513794, 0.14252454042434692, 0.49773427844047546, -0.42716190218925476, 0.03323405608534813, -0.3867594301700592, -0.0784095898270607, 0.13172432780265808, 0.3587783873081207, 0.5109906196594238, 0.49511265754699707, 0.7409348487854004, 0.21091100573539734, -0.3797110915184021, -0.12114308774471283
                    };
                    for (int i = 0 ; i < queryVector1.length ; ++i) {
                        queryVector[i] = (float) queryVector1[i];
                    }
                    TopKnnCollector collector = new TopKnnCollector(TOP_K * 2, Integer.MAX_VALUE);
                    searcher.search(queryVector, collector, AcceptDocs.fromLiveDocs(null, vectors.length));
                    // Result is discarded — this is for breakpoint debugging
                    TopDocs results = collector.topDocs();
                    for (ScoreDoc scoreDoc : results.scoreDocs) {
                        System.out.println("Doc ID: " + scoreDoc.doc + ", Score: " + scoreDoc.score);
                    }

                    // Rerank test: build doc-to-score map from ANN results, then rerank with error residual
                    System.out.println("\nRerank results:");
                    java.util.Map<Integer, Float> docIdToScore = new java.util.HashMap<>();
                    for (ScoreDoc scoreDoc : results.scoreDocs) {
                        docIdToScore.put(scoreDoc.doc, scoreDoc.score);
                    }

                    // Get a fresh BBQ reader to obtain OffHeapBinarizedVectorValues for reranking
                    FieldInfos rerankFieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
                    SegmentReadState rerankReadState = new SegmentReadState(
                        directory, segmentInfo, rerankFieldInfos, IOContext.DEFAULT, "test_field"
                    );
                    try (BBQReader rerankReader = new BBQReader(rerankReadState, new Lucene102BinaryFlatVectorsScorer())) {
                        FloatVectorValues rerankFloatValues = rerankReader.getFloatVectorValues("test_field");
                        BBQReader.BinarizedVectorValues rerankBinarized = (BBQReader.BinarizedVectorValues) rerankFloatValues;
                        OffHeapBinarizedVectorValues offHeapValues =
                            (OffHeapBinarizedVectorValues) rerankBinarized.quantizedVectorValues;

                        offHeapValues.initRerank(queryVector);

                        // Rerank each candidate and compare with ground truth
                        System.out.printf("%-8s %-14s %-14s %-14s %-14s %-14s %-14s%n", "DocID", "ApproxMIP", "RerankedMIP", "TrueMIP", "TrueDot", "RerankError", "ApproxError");
                        for (ScoreDoc scoreDoc : results.scoreDocs) {
                            int docId = scoreDoc.doc;
                            float approxMip = scoreDoc.score;
                            float rerankedMip = offHeapValues.rerank(docId, approxMip);

                            // Ground truth
                            float trueDot = 0f;
                            for (int d = 0; d < dimension; d++) {
                                trueDot += queryVector[d] * vectors[docId][d];
                            }
                            final float trueMip = VectorUtil.scaleMaxInnerProductScore(trueDot);

                            System.out.printf("%-8d %-14.6f %-14.6f %-14.6f %-14.6f %-14.6f %-14.6f%n", docId, approxMip, rerankedMip, trueMip, trueDot, trueMip - rerankedMip, trueMip - approxMip);
                        }
                    }
                }
            }
        }

        System.out.println();
    }

    private void writeVectors(Directory directory, float[][] vectors, SegmentInfo segmentInfo, FieldInfo fieldInfo) throws IOException {
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        SegmentWriteState writeState = new SegmentWriteState(
            InfoStream.NO_OUTPUT, directory, segmentInfo, fieldInfos, null, IOContext.DEFAULT, "test_field"
        );

        Lucene102BinaryFlatVectorsScorer scorer = new Lucene102BinaryFlatVectorsScorer();
        try (BBQWriter writer = new BBQWriter(scorer, writeState)) {
            FlatFieldVectorsWriter fieldWriter = writer.addField(fieldInfo);
            for (int i = 0; i < vectors.length; i++) {
                // TMP
                if (i == 385) {
                    System.out.println();
                }
                // TMP
                fieldWriter.addValue(i, vectors[i].clone());
            }
            writer.flush(vectors.length, null);
            writer.finish();
        }
    }

    private long ingestIntoFaiss(Directory directory, int dimension, int numVectors, SegmentInfo segmentInfo, FieldInfo fieldInfo) throws IOException {
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        SegmentReadState readState = new SegmentReadState(
            directory, segmentInfo, fieldInfos, IOContext.DEFAULT, "test_field"
        );

        Lucene102BinaryFlatVectorsScorer scorer = new Lucene102BinaryFlatVectorsScorer();
        try (BBQReader reader = new BBQReader(readState, scorer)) {
            FloatVectorValues floatVectorValues = reader.getFloatVectorValues("test_field");
            BBQReader.BinarizedVectorValues binarizedVectorValues = (BBQReader.BinarizedVectorValues) floatVectorValues;
            BinarizedByteVectorValues quantizedVectorValues = binarizedVectorValues.quantizedVectorValues;

            int quantizedVecBytes = quantizedVectorValues.vectorValue(0).length;
            float centroidDp = quantizedVectorValues.getCentroidDP();

            // Initialize Faiss BBQ index
            Map<String, Object> parameters = new HashMap<>();
            parameters.put("name", "hnsw");
            parameters.put("data_type", "float");
            parameters.put("index_description", "BHNSW16,Flat");
            parameters.put("spaceType", "innerproduct");
            Map<String, Object> subParameters = new HashMap<>();
            parameters.put("parameters", subParameters);
            subParameters.put("ef_search", 256);
            subParameters.put("ef_construction", 256);
            subParameters.put("m", 16);
            subParameters.put("encoder", Collections.emptyMap());
            subParameters.put("indexThreadQty", 1);

            long indexMemoryAddress = FaissService.initBBQIndex(
                numVectors, dimension, parameters, centroidDp, quantizedVecBytes
            );

            // Pass quantized vectors + correction factors in batches
            passQuantizedVectors(indexMemoryAddress, binarizedVectorValues);

            // Add doc IDs
            int batchSize = 500;
            int[] docIds = new int[batchSize];
            int numAdded = 0;
            int remaining = numVectors;
            while (remaining > 0) {
                int count = Math.min(batchSize, remaining);
                for (int i = 0; i < count; i++) {
                    docIds[i] = numAdded + i;
                }
                FaissService.addDocsToBBQIndex(indexMemoryAddress, docIds, count, numAdded);
                numAdded += count;
                remaining -= count;
            }

            return indexMemoryAddress;
        }
    }

    private void passQuantizedVectors(
        final long indexMemoryAddress,
        final BBQReader.BinarizedVectorValues binarizedVectorValues
    ) throws IOException {
        final int batchSize = 500;
        byte[] buffer = null;
        for (int i = 0; i < binarizedVectorValues.size(); ) {
            final int loopSize = Math.min(binarizedVectorValues.size() - i, batchSize);
            for (int j = 0, o = 0; j < loopSize; ++j) {
                final byte[] binaryVector = binarizedVectorValues.quantizedVectorValues.vectorValue(i + j);
                if (buffer == null) {
                    // [Quantized Vector | lowerInterval (float) | upperInterval (float) | additionalCorrection (float) | quantizedComponentSum (int)]
                    buffer = new byte[(binaryVector.length + Integer.BYTES * 4) * batchSize];
                }
                final OptimizedScalarQuantizer.QuantizationResult quantizationResult =
                    binarizedVectorValues.quantizedVectorValues.getCorrectiveTerms(i + j);

                // Copy quantized vector
                System.arraycopy(binaryVector, 0, buffer, o, binaryVector.length);
                o += binaryVector.length;

                // Copy correction factors
                int bits = Float.floatToRawIntBits(quantizationResult.lowerInterval());
                buffer[o++] = (byte) (bits);
                buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16);
                buffer[o++] = (byte) (bits >>> 24);

                bits = Float.floatToRawIntBits(quantizationResult.upperInterval());
                buffer[o++] = (byte) (bits);
                buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16);
                buffer[o++] = (byte) (bits >>> 24);

                bits = Float.floatToRawIntBits(quantizationResult.additionalCorrection());
                buffer[o++] = (byte) (bits);
                buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16);
                buffer[o++] = (byte) (bits >>> 24);

                bits = quantizationResult.quantizedComponentSum();
                buffer[o++] = (byte) (bits);
                buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16);
                buffer[o++] = (byte) (bits >>> 24);
            }

            FaissService.passBBQVectors(indexMemoryAddress, buffer, loopSize);

            i += loopSize;
        }
    }
}
