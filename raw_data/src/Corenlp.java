package Corenlp;

/**
*Extracting keywords from String inputs or local text files
@author Nan Ding
*/

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.TreeMap;

import com.clearspring.analytics.util.Pair;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.TypedDependency;
import edu.stanford.nlp.util.CoreMap;

public class Corenlp {

	private static StanfordCoreNLP pipeline;
	private static List<String> stopWordList;
	
    /**
     * Test keywordExtraction
     * @param args
     */
    public static void main(String[] args)
    {
    	long startTime = System.nanoTime();
    	
    	Corenlp key = new Corenlp();
        String text = "Professor Emerson gave the best presentation of machine learning!";
        List<Pair<String, Integer>> result = keywordExtraction(text);
        System.out.println(result);
        
    	long endTime = System.nanoTime();
    	System.out.println("Took "+ (endTime - startTime)/1000000000 + '.' + (endTime - startTime)%1000000000 + " ns"); 
    }

    public Corenlp()
    {
    	Properties props = new Properties();    // creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma");    //props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
        this.pipeline = new StanfordCoreNLP(props);
        this.stopWordList = readStopWord();
    }
    
    /**
     * get <keyword, frequency> map by input string
     * @param emailText
     * @return
     */
    public static List<Pair<String, Integer>> keywordExtraction(String emailText)
    {
        String text = emailText;                		// read text in the text variable
        Annotation document = new Annotation(text);    	// create an empty Annotation just with the given text
        pipeline.annotate(document);            		// run all Annotators on this text
        List<CoreMap> sentences = document.get(SentencesAnnotation.class);
        Map<String, Integer> result = new LinkedHashMap<>();    // output Hash map
        // extract single keyword
        for(CoreMap sentence: sentences) {
            for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
                String word = token.get(TextAnnotation.class);                   // this is the text of the token
                String pos = token.get(PartOfSpeechAnnotation.class);            // this is the POS tag of the token
                String ne = token.get(NamedEntityTagAnnotation.class);           // this is the NER label of the token
                String lemma = token.get(LemmaAnnotation.class).toLowerCase();   // this is the Lemma label of the token
//                System.out.println(word + '\t' + pos + '\t' + ne + '\t' + lemma);
                // filtering by part-of-speech tag and name-entity-recognition tag
                if(!stopWordList.contains(lemma) && !lemma.matches("^.*[^a-zA-Z ].*$"))
                {
                        if(result.containsKey(lemma))
                            result.put(lemma, result.get(lemma) + 1);
                        else
                            result.put(lemma, 1);
                }
            }
        }

        // transfer result
        List<Pair<String, Integer>> rest = new ArrayList<Pair<String, Integer>>();
        for(Entry<String, Integer> it: result.entrySet())
        {
        	Pair<String, Integer> row = new Pair<String, Integer>(it.getKey(), it.getValue());
        	rest.add(row);
        }
        // return result
        return rest;
    }

    /**
     * Read stop word list from local directory
     * @return
     */
    public static List<String> readStopWord()
    {
        List<String> stopWordList = new ArrayList<String>();
        String csvFile = "bin/StopWord.csv";		// open local stop word list file
        BufferedReader br = null;
        String line = "";
        try {
            br = new BufferedReader(new FileReader(csvFile));
            while ((line = br.readLine()) != null) {
                stopWordList.add(line);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null)
            {
                try{
                    br.close();}
                catch (IOException e) {
                    e.printStackTrace();}
            }
        }
        return stopWordList;
    }

    /**
     * Print <keyword, frequency> Map
     * @param result
     */
    public static void printResult(Map<String, Integer> result)
    {
        Iterator it = result.entrySet().iterator();
        while(it.hasNext())
        {
            Map.Entry pair = (Map.Entry)it.next();
            System.out.println(pair.getKey() + " = " + pair.getValue());
        }
    }
}