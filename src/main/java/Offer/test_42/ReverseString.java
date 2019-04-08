package Offer.test_42;

public class ReverseString {
    public static void main(String[] args) {
//        String str = "I am a student.";
//        String str = null;
//        String str = "    ";
        String str = "tantand";
//        String str = "a";
        System.out.println(reverseSequence(str));

//        System.out.println(str.length()+" "+str.toCharArray().length);
    }

    public static String reverseSequence(String str){
        if(str == null || str.length() == 0)
            return null;
        char[] chars = str.toCharArray();
        //翻转整个句子
        reverse(chars, 0, chars.length - 1);
        int i = 0, j = 0;
        while(j <= chars.length){
            if(j == str.length() || chars[j] == ' '){
                reverse(chars, i, j - 1);
                i = j + 1;
            }
            j++;
        }
//        return chars.toString();
        return new String(chars);
    }

    public static void reverse(char[] chars, int i, int j){
        if(chars == null || chars.length == 0)
            return;
        while(i < j){
            char ch = chars[i];
            chars[i] = chars[j];
            chars[j] = ch;

            i++;
            j--;
        }
    }
}
