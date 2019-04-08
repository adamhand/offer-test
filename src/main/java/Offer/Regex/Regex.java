package Offer.Regex;
import java.util.Scanner;

//正则表达式匹配
public class Regex {
    public static boolean match(String input,String pattern){
        if(input==null||pattern==null) return false;
        return matchCore(input,0,pattern,0);
    }
    private static boolean matchCore(String input,int i,String pattern,int p){
        if((input.length()==i)&&(pattern.length()==p)){
            //出口1，input和pattern都到了字符串末尾
            return true;
        }
        if((i!=input.length())&&(pattern.length()==p)){
            //出口2，字符串input没有到末尾，pattern到了末尾
            return false;
        }
        if((input.length()==i)&&(pattern.length()!=p)){
            //出口3，字符串input到末尾，pattern还没有到末尾
            return false;
        }

        if((p+1<pattern.length())&&(pattern.charAt(p+1)=='*')){//pattern第二个字符为*
            if((input.charAt(i)==pattern.charAt(p))||(pattern.charAt(p)=='.')){
                //首字母相匹配
                return matchCore(input,i+1,pattern,p+2) //*表示出现1次
                        ||matchCore(input,i+1,pattern,p)    //*表示出现多次
                        ||matchCore(input,i,pattern,p+2);   //*表示出现0次 ， a ...  p* ...
            }else{
                //首字母不匹配
                return matchCore(input,i,pattern,p+2);
            }
        } //end pattern.charAt(p+1)=='*'

        if((input.charAt(i)==pattern.charAt(p))||(pattern.charAt(p)=='.')){
            //pattern第二个字母不是*，且首字母匹配
            return matchCore(input,i+1,pattern,p+1);
        }
        return false;  //其余情况全部不匹配
    }

    public static void main(String[] args) {
        // TODO Auto-generated method stub
        Scanner scanner = new Scanner(System.in);                       //扫描键盘输入
        System.out.println(" 请输入第一个字符串：");
        String str1 = scanner.nextLine();
        System.out.println(" 请输入第二个字符串：");
        String str2 = scanner.nextLine();
        scanner.close();

        System.out.print("匹配的结果为：");
        System.out.println(match(str1, str2));
    }

}
