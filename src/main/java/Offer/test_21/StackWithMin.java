package Offer.test_21;

import java.util.Stack;

//包含min函数的栈
public class StackWithMin {
    private Stack<Integer> dataStack = new Stack<>();
    private Stack<Integer> minStack = new Stack<>();

    public void push(int data){
        dataStack.push(data);
        minStack.push(minStack.isEmpty() ? data : Math.min(data, minStack.peek()));
    }

    public void pop(){
        if(!dataStack.isEmpty() && !minStack.isEmpty()){
            dataStack.pop();
            minStack.pop();
        }
    }

    public int top(){
        if(!dataStack.isEmpty())
            return dataStack.peek();
        return -1;
    }

    public int min(){
        if(!minStack.isEmpty())
            return minStack.peek();
        return -1;
    }
}
