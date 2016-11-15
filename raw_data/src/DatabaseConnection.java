package Corenlp;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

public class DatabaseConnection {
	protected Map<String, String> path = new HashMap<String, String>();
	
	/**
	 * connect input database
	 */
	DatabaseConnection()
	{
		try{
			FileReader fr = new FileReader("src/Corenlp/DBConnection.txt");
			BufferedReader br = new BufferedReader(fr);
			path.put("url", br.readLine());
			path.put("user", br.readLine());
			path.put("password", br.readLine());
		}
		catch (Exception ex){
			Logger.getLogger(DatabaseConnection.class.getName()).log(Level.SEVERE, null, ex);
		}
	}
}
