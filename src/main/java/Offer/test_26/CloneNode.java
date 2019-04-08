package Offer.test_26;

class RandomListNode{
    int value;
    RandomListNode next;
    RandomListNode random;

    public RandomListNode(int value) {
        this.value = value;
    }
}

//复制复杂链表的节点
public class CloneNode {
    public static void main(String[] args) {
        Test.Test1();
//        Test.Test2();
//        Test.Test3();
//        Test.Test4();
//        Test.Test5();
    }

    /**'
     * 方法一：使用HashMap存放对应关系
     * @param root
     * @return
     */
//    public static RandomListNode cloneRandomList(RandomListNode root){
//        if(root == null)
//            return null;
//        HashMap<RandomListNode, RandomListNode> map = new HashMap<>();
//        RandomListNode cloneRoot = new RandomListNode(root.value);
//        RandomListNode node = root, cloneNode = cloneRoot;
//        map.put(node, cloneNode);
//
//        while(node.next != null){
//            cloneNode.next = new RandomListNode(node.next.value);
//            node = node.next;
//            cloneNode = cloneNode.next;
//            map.put(node, cloneNode);
//        }
//
//        node = root;
//        cloneNode = cloneRoot;
//        while(cloneNode != null){
//            cloneNode.random = map.get(node.random);
//            node = node.next;
//            cloneNode = cloneNode.next;
//        }
//        return cloneRoot;
//    }

    /**
     * 方法二：先复制在拆分
     */
    public static RandomListNode cloneRandomList(RandomListNode pHead) {
        if (pHead == null)
            return null;
        // 插入新节点
        RandomListNode cur = pHead;
        while (cur != null) {
            RandomListNode clone = new RandomListNode(cur.value);
            clone.next = cur.next;
            cur.next = clone;
            cur = clone.next;
        }
        // 建立 random 链接
        cur = pHead;
        while (cur != null) {
            RandomListNode clone = cur.next;
            if (cur.random != null)
                clone.random = cur.random.next;
            cur = clone.next;
        }
        // 拆分
        cur = pHead;
        RandomListNode pCloneHead = pHead.next;
        while (cur.next != null) {
            RandomListNode next = cur.next;
            cur.next = next.next;
            cur = next;
        }
        return pCloneHead;
    }


    //链表公共方法：创建、输出等等
    static class RandomListUtils{
        public static RandomListNode createList(int val){
            RandomListNode node = new RandomListNode(val);
            node.next = null;
            node.random = null;
            return node;
        }

        public static void buildNode(RandomListNode node, RandomListNode next, RandomListNode random){
            if(node != null){
                node.next = next;
                node.random = random;
            }
        }

        public static void printList(RandomListNode root){
            RandomListNode node = root;
            while(node != null){
                System.out.println("The value of this node is:"+node.value);
                if(node.random != null)
                    System.out.println("The value of its sibling is:"+node.random.value);
                else
                    System.out.println("This node does not have a sibling.");
                System.out.println();
                node = node.next;
            }
        }
    }

    //测试类
    static class Test{
        public static void test(String testName, RandomListNode root){
            if(testName != null)
                System.out.println(testName+"begins:");
            System.out.println("The original list is:");
            RandomListUtils.printList(root);

            RandomListNode cloneRoot = cloneRandomList(root);
            System.out.println("The cloned list is:");
            RandomListUtils.printList(cloneRoot);
        }

        //          -----------------
        //         \|/              |
        //  1-------2-------3-------4-------5
        //  |       |      /|\             /|\
        //  --------+--------               |
        //          -------------------------
        public static void Test1()
        {
            RandomListNode pNode1 = RandomListUtils.createList(1);
            RandomListNode pNode2 = RandomListUtils.createList(2);
            RandomListNode pNode3 = RandomListUtils.createList(3);
            RandomListNode pNode4 = RandomListUtils.createList(4);
            RandomListNode pNode5 = RandomListUtils.createList(5);

            RandomListUtils.buildNode(pNode1, pNode2, pNode3);
            RandomListUtils.buildNode(pNode2, pNode3, pNode5);
            RandomListUtils.buildNode(pNode3, pNode4, null);
            RandomListUtils.buildNode(pNode4, pNode5, pNode2);

            test("Test1", pNode1);
        }

        // m_pSibling指向结点自身
        //          -----------------
        //         \|/              |
        //  1-------2-------3-------4-------5
        //         |       | /|\           /|\
        //         |       | --             |
        //         |------------------------|
        public static void Test2()
        {
            RandomListNode pNode1 = RandomListUtils.createList(1);
            RandomListNode pNode2 = RandomListUtils.createList(2);
            RandomListNode pNode3 = RandomListUtils.createList(3);
            RandomListNode pNode4 = RandomListUtils.createList(4);
            RandomListNode pNode5 = RandomListUtils.createList(5);

            RandomListUtils.buildNode(pNode1, pNode2, null);
            RandomListUtils.buildNode(pNode2, pNode3, pNode5);
            RandomListUtils.buildNode(pNode3, pNode4, pNode3);
            RandomListUtils.buildNode(pNode4, pNode5, pNode2);

            test("Test2", pNode1);
        }

        // m_pSibling形成环
        //          -----------------
        //         \|/              |
        //  1-------2-------3-------4-------5
        //          |              /|\
        //          |               |
        //          |---------------|
        public static void Test3()
        {
            RandomListNode pNode1 = RandomListUtils.createList(1);
            RandomListNode pNode2 = RandomListUtils.createList(2);
            RandomListNode pNode3 = RandomListUtils.createList(3);
            RandomListNode pNode4 = RandomListUtils.createList(4);
            RandomListNode pNode5 = RandomListUtils.createList(5);

            RandomListUtils.buildNode(pNode1, pNode2, null);
            RandomListUtils.buildNode(pNode2, pNode3, pNode4);
            RandomListUtils.buildNode(pNode3, pNode4, null);
            RandomListUtils.buildNode(pNode4, pNode5, pNode2);

            test("Test3", pNode1);
        }

        // 只有一个结点
        public static void Test4()
        {
            RandomListNode pNode1 = RandomListUtils.createList(1);
            RandomListUtils.buildNode(pNode1, null, pNode1);

            test("Test4", pNode1);
        }

        // 鲁棒性测试
        public static void Test5()
        {
            test("Test5", null);
        }
    }
}
