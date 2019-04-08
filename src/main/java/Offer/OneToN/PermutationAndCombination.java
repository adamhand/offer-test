package Offer.OneToN;

import java.util.ArrayList;
import java.util.List;

//排列组合问题
public class PermutationAndCombination {
    public static void permutation(char[] array, int index){
        if(index==array.length){
            System.out.println(array);
            return;
        }

        if(array.length==0||index<0||index>array.length){
            return;
        }

        for(int j=index;j<array.length;j++){
            char temp=array[j];
            array[j]=array[index];
            array[index]=temp;
            permutation(array, index+1);
            temp=array[j];
            array[j]=array[index];
            array[index]=temp;
        }
    }

    public static void combiantion(char chs[]){
        if(chs==null||chs.length==0){
            return ;
        }
        List<Character> list=new ArrayList();
        for(int i=1;i<=chs.length;i++){
            combine(chs,0,i,list);
        }
    }
    //从字符数组中第begin个字符开始挑选number个字符加入list中
    public static void combine(char []cs,int begin,int number,List<Character> list){
        if(number==0){
            System.out.println(list.toString());
            return ;
        }
        if(begin==cs.length){
            return;
        }
        list.add(cs[begin]);
        combine(cs,begin+1,number-1,list);
        list.remove((Character)cs[begin]);
        combine(cs,begin+1,number,list);
    }

    public static void main(String[] args) {
        char[] chars={'a','b','c'};
        permutation(chars,0);
        combiantion(chars);
    }
}
