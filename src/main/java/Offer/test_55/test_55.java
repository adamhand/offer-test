package Offer.test_55;

import java.util.ArrayList;
import java.util.HashMap;

public class test_55 {
    static HashMap<Character, Integer> map = new HashMap<>();
    static ArrayList<Character> list = new ArrayList<>();

    public static void main(String[] args) {
        char[] chars = "google".toCharArray();
        for(int i = 0; i < chars.length; i++)
            insert(chars[i]);
        System.out.println(firstAppearOnce());
    }

    public static char firstAppearOnce(){
        for(char key : list){
            if(map.get(key) == 1)
                return key;
        }
        return '#';
    }

    public static void insert(char ch){
        if(map.containsKey(ch))
            map.put(ch, map.get(ch)+1);
        else
            map.put(ch, 1);
        list.add(ch);
    }
}
