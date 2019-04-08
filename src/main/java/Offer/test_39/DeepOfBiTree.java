package Offer.test_39;

import DataStructureAndAlgorithm.BiTree2.BiTreeNode;

//二叉搜索树的深度
public class DeepOfBiTree {
    public static void main(String[] args) {

    }

    public static int biTreeDeepth(BiTreeNode root){
        if(root == null)
            return -1;
        int nLeft = biTreeDeepth(root.getLeft());
        int nRight = biTreeDeepth(root.getRight());

        return (nLeft > nRight) ? (nLeft + 1) : (nRight + 1);
    }
}
