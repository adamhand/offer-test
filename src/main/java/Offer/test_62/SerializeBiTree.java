package Offer.test_62;

import DataStructureAndAlgorithm.BiTree2.BiTreeNode;

import java.util.Stack;

public class SerializeBiTree {
    public static void main(String[] args) {

    }

    /**
     * 解法一：序列化与反序列化
     * @param root
     * @param sb
     */
    //递归法序列化
    private static void serializeBiTree(BiTreeNode root, StringBuilder sb){
        if(root == null){
            sb.append("# ");
            return;
        }
        sb.append(root.getValue());
        serializeBiTree(root.getLeft(), sb);
        serializeBiTree(root.getRight(), sb);
    }
    public static String serializeBiTree(BiTreeNode root){
        StringBuilder sb = new StringBuilder("");
        serializeBiTree(root, sb);
        return sb.toString();
    }

    //非递归法序列化
    private static void nrSerializeBiTree(BiTreeNode root, StringBuilder sb){
        if(root == null)
            return;

        Stack<BiTreeNode> stack = new Stack<>();
        BiTreeNode pNode = root;
        stack.push(pNode);
        while(!stack.isEmpty()){
            BiTreeNode temp = stack.pop();
            if(temp == null)
                sb.append("# ");
            else {
                sb.append(temp.getValue()).append(" ");
                nrSerializeBiTree(root.getLeft(), sb);
                nrSerializeBiTree(root.getRight(), sb);
            }
        }
    }

    //递归法反序列化
    private String deserializeStr;
    public BiTreeNode deserialize(String str) {
        deserializeStr = str;
        return deserialize();
    }

    private BiTreeNode deserialize() {
        if (deserializeStr.length() == 0)
            return null;
        int index = deserializeStr.indexOf(" ");
        String node = index == -1 ? deserializeStr : deserializeStr.substring(0, index);
        deserializeStr = index == -1 ? "" : deserializeStr.substring(index + 1);
        if (node.equals("#"))
            return null;
        int val = Integer.valueOf(node);
        BiTreeNode t = new BiTreeNode(val);
        t.setLeft(deserialize());
        t.setRight(deserialize());
        return t;
    }
}
