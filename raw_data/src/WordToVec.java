/**
 * This class create sparse matrix of each review
 * create one-hot vector of all reviews
 * create dictionary of one-hot vector
 * connect to database
 */

package Corenlp;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.clearspring.analytics.util.Pair;
import com.fasterxml.jackson.core.Version;

public class WordToVec {
	public static void main(String[] args) throws SQLException
	{
    	long startTime = System.nanoTime();
    	
		WordToVec t = new WordToVec();
		List<Pair<Pair<String, String>, List<Pair<String, Integer>>>> reviews = t.singleReviewToVector();	// create review vector
		Map<String, Integer> oneHotMap = t.createOneHotMap(reviews);	// create oneHot vector
		t.createSingleHotReviewsSparseMatrix(reviews, oneHotMap);		// create sparse matrix
		t.createDict(oneHotMap);										// create dict
    	long endTime = System.nanoTime();
    	System.out.println("Took "+ (endTime - startTime)/1000000000 + '.' + (endTime - startTime)%1000000000 + " ns"); 
	}
	
	/**
	 * define database connection variables
	 */
	DatabaseConnection dbConn = new DatabaseConnection();
	public final String url = dbConn.path.get("url");
	public final String user = dbConn.path.get("user");
	public final String password = dbConn.path.get("password");
	public Connection con = null;
	public Statement st = null;
	public ResultSet rs = null;
	
	public WordToVec()
	{
		try
		{
			con = DriverManager.getConnection(url, user, password);
			st = con.createStatement();
		}
		catch (Exception ex)
		{
			Logger lgr = Logger.getLogger(Version.class.getName());
            lgr.log(Level.SEVERE, ex.getMessage(), ex);
		}
	}
	
	/**
	 * get result vector, one-hot vector
	 * transfer review vector into sparse matrix
	 * write result as file
	 */
	public void createSingleHotReviewsSparseMatrix(List<Pair<Pair<String, String>, List<Pair<String, Integer>>>> reviews, Map<String, Integer> singleHotMap)
	{
		Path file = Paths.get("/Users/emerson/Documents/yelp/Reviews_Sparse_Top_Users_Business.csv");
		List<String> result = new ArrayList<String>();
		int count = 0;
		for(Pair<Pair<String, String>, List<Pair<String, Integer>>> row: reviews)
		{
			String rowReviewVector = row.left.left + ',' + row.left.right + ',';
			for(Pair<String, Integer> line: row.right)
			{
				rowReviewVector += singleHotMap.get(line.left).toString() + ":"+ line.right.toString() + " ";
			}
			++count;
			result.add(rowReviewVector);
			System.out.print(rowReviewVector);
			System.out.println();
//			row = null;
//			rowReviewVector = null;
			if(count == 5000)
			{
				count = 0;
				try {
					Files.write(file, result, Charset.forName("UTF-8"));
					result.clear();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
		if(count != 0)
		{
			try {
				Files.write(file, result, Charset.forName("UTF-8"));
				result.clear();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * get review vectors and one-hot vector
	 * represent review vector as one-hot vector
	 * write result as file
	 */
	public void createSingleHotReviews(List<Pair<Pair<String, String>, List<Pair<String, Integer>>>> reviews, Map<String, Integer> singleHotMap)
	{
		Path file = Paths.get("/Users/emerson/Documents/yelp/Reviews_Top_Users.csv");
		List<String> result = new ArrayList<String>();
		int count = 0;
		for(Pair<Pair<String, String>, List<Pair<String, Integer>>> row: reviews)
		{
			Map<String, Integer> rowMap = new TreeMap<String, Integer>(singleHotMap);
			for(Pair<String, Integer> line: row.right)
			{
				rowMap.put(line.left, line.right);
			}
			
			String rowReviewVector = row.left.left + ',' + row.left.right + ',';
			for(Entry<String, Integer> it: rowMap.entrySet())
			{
				rowReviewVector += it.getValue().toString();
			}
			++count;
			result.add(rowReviewVector);
			System.out.print(rowReviewVector);
			System.out.println();
//			row = null;
//			rowReviewVector = null;
			if(count == 5000)
			{
				count = 0;
				try {
					Files.write(file, result, Charset.forName("UTF-8"));
					result.clear();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}
	
	/**
	 * get reviews from database
	 * use keyWordExtraction to get keywords from reviews
	 * return keyword vector of reviews
	 */
	public List<Pair<Pair<String, String>, List<Pair<String, Integer>>>> singleReviewToVector() throws SQLException
	{
		List<Pair<Pair<String, String>, List<Pair<String, Integer>>>> result = new ArrayList<Pair<Pair<String, String>, List<Pair<String, Integer>>>>();
		Corenlp keyWordExtraction = new Corenlp();		// object to split words
		getReviews();		// get reviews from db
//		int count = 0;		
		while(rs.next())
		{
			Pair<String, String> key = new Pair<String, String>(rs.getString(1), rs.getString(2));
			List<Pair<String, Integer>> words = new ArrayList<Pair<String, Integer>>();
			words = keyWordExtraction.keywordExtraction(rs.getString(3));
			Pair<Pair<String, String>, List<Pair<String, Integer>>> row = new Pair<Pair<String, String>, List<Pair<String, Integer>>>(key, words);
			result.add(row);
//			System.out.println(row);
//			System.out.println(count++);
		}
		return result;
	}
	
	/**
	 * get reviews vector
	 * count distinct word of all reviews and put them in order
	 * return a TreeMap with distinct words
	 */
	public Map<String, Integer> createOneHotMap(List<Pair<Pair<String, String>, List<Pair<String, Integer>>>> reviews)
	{
		Map<String, Integer> result = new TreeMap<String, Integer>();
		for(Pair<Pair<String, String>, List<Pair<String, Integer>>> row: reviews)
		{
			for(Pair<String, Integer> it: row.right)
			{
				if(!result.containsKey(it.left))
					result.put(it.left, 0);
			}
		}
		// set index of Map. Need to be commented for non sparse matrix
		int count = 0;
		for(Entry<String, Integer> row: result.entrySet())
		{
			row.setValue(count++);
		}
		System.out.println(result);
		return result;
	}
	
	/**
	 * get onehotmap
	 * write as file
	 */
	public void createDict(Map<String, Integer> oneHotMap)
	{
		Path file = Paths.get("/Users/emerson/Documents/yelp/Dict.csv");
		List<String> result = new ArrayList<String>();
		int count = 0;
		for(Map.Entry<String, Integer> line: oneHotMap.entrySet())
			result.add(count++ + "\t" + line.getKey());
		try {
			Files.write(file, result, Charset.forName("UTF-8"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * get reviews from database
	 */
	protected void getReviews()
	{
		try {
			rs = st.executeQuery(queryReviews());
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * get reviews with user commented more than 50 and business commented more than 100 from database
	 */
	protected String queryReviews()
	{
		return "SELECT "
			   +"two.user_id, o.business_id, text FROM "
			   +"yelp.yelp_academic_dataset_review AS o "
			   +"RIGHT JOIN "
			   +"(SELECT "
			   +"user_id "
			   +"FROM "
			   +"(SELECT "
			   +"user_id, business_id, stars "
			   +"FROM "
			   +"yelp.yelp_academic_dataset_review AS a "
			   +"WHERE "
			   +"user_id NOT LIKE '% %') AS a "
			   +"GROUP BY user_id "
			   +"HAVING COUNT(*) > 50 "
			   +"ORDER BY COUNT(*) DESC) AS two ON o.user_id = two.user_id "
			   +"RIGHT JOIN "
			   +"(SELECT "
			   +"business_id "
			   +"FROM "
			   +"yelp.yelp_academic_dataset_review "
			   +"GROUP BY business_id "
			   +"HAVING COUNT(*) > 100) AS three ON o.business_id = three.business_id "
			   +"where two.user_id <> '' "
			   +"ORDER BY two.user_id";
	}
}
