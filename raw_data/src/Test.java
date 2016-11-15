package Corenlp;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Test {
	public static void main(String[] args) throws IOException
	{
		Map<String, Integer> map = new HashMap<String, Integer>();
		map.put("a", 1);
		Map<String, Integer> map2 = map;
		map.remove("a");
		System.out.print(map2.get("a"));
	}
}
