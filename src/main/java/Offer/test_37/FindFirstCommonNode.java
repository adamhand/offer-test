package Offer.test_37;

import Offer.ListNodeUtils.ListNode;

class ListUtils{
    public static ListNode createListNode(int val){
        ListNode node = new ListNode(val);
        node.next = null;

        return node;
    }

    public static void connectListNode(ListNode cur, ListNode next){
        if(cur == null)
            return;
        cur.next = next;
    }
}

//链表第一个相同的节点
public class FindFirstCommonNode {
    public static void main(String[] args) {
        test1();
        test2();
        test3();
        test4();
        test5();
        test6();
    }

    public static void test(String testName, ListNode head1, ListNode head2, ListNode expected){
        if(testName != null)
            System.out.println(testName+" begins:");
        ListNode result = findFirstCommonNode(head1, head2);
//        System.out.println(result.value);
        if(result == expected)
            System.out.println("Passed");
        else
            System.out.println("Failed");
        System.out.println();
    }

    // 第一个公共结点在链表中间
    // 1 - 2 - 3 \
    //            6 - 7
    //     4 - 5 /
    public static void test1(){
        ListNode node1 = ListUtils.createListNode(1);
        ListNode node2 = ListUtils.createListNode(2);
        ListNode node3 = ListUtils.createListNode(3);
        ListNode node4 = ListUtils.createListNode(4);
        ListNode node5 = ListUtils.createListNode(5);
        ListNode node6 = ListUtils.createListNode(6);
        ListNode node7 = ListUtils.createListNode(7);

        ListUtils.connectListNode(node1, node2);
        ListUtils.connectListNode(node2, node3);
        ListUtils.connectListNode(node3, node6);
        ListUtils.connectListNode(node4, node5);
        ListUtils.connectListNode(node5, node6);
        ListUtils.connectListNode(node6, node7);

//        ListNodeUtils.printList(node1);
        test("test1", node1, node4, node6);
    }

    // 没有公共结点
    // 1 - 2 - 3 - 4
    //
    // 5 - 6 - 7
    public static void test2(){
        ListNode node1 = ListUtils.createListNode(1);
        ListNode node2 = ListUtils.createListNode(2);
        ListNode node3 = ListUtils.createListNode(3);
        ListNode node4 = ListUtils.createListNode(4);
        ListNode node5 = ListUtils.createListNode(5);
        ListNode node6 = ListUtils.createListNode(6);
        ListNode node7 = ListUtils.createListNode(7);

        ListUtils.connectListNode(node1, node2);
        ListUtils.connectListNode(node2, node3);
        ListUtils.connectListNode(node3, node4);
        ListUtils.connectListNode(node5, node6);
        ListUtils.connectListNode(node6, node7);

//        ListNodeUtils.printList(node1);
        test("test2", node1, node5, null);
    }

    // 公共结点是最后一个结点
    // 1 - 2 - 3 - 4 \
    //                7
    //         5 - 6 /
    public static void test3(){
        ListNode node1 = ListUtils.createListNode(1);
        ListNode node2 = ListUtils.createListNode(2);
        ListNode node3 = ListUtils.createListNode(3);
        ListNode node4 = ListUtils.createListNode(4);
        ListNode node5 = ListUtils.createListNode(5);
        ListNode node6 = ListUtils.createListNode(6);
        ListNode node7 = ListUtils.createListNode(7);

        ListUtils.connectListNode(node1, node2);
        ListUtils.connectListNode(node2, node3);
        ListUtils.connectListNode(node3, node4);
        ListUtils.connectListNode(node4, node7);


        ListUtils.connectListNode(node5, node6);
        ListUtils.connectListNode(node6, node7);

//        ListNodeUtils.printList(node1);
        test("test3", node1, node5, node7);
    }

    // 公共结点是第一个结点
    // 1 - 2 - 3 - 4 - 5
    // 两个链表完全重合
    public static void test4(){
        ListNode node1 = ListUtils.createListNode(1);
        ListNode node2 = ListUtils.createListNode(2);
        ListNode node3 = ListUtils.createListNode(3);
        ListNode node4 = ListUtils.createListNode(4);
        ListNode node5 = ListUtils.createListNode(5);


        ListUtils.connectListNode(node1, node2);
        ListUtils.connectListNode(node2, node3);
        ListUtils.connectListNode(node3, node4);
        ListUtils.connectListNode(node4, node5);

//        ListNodeUtils.printList(node1);
        test("test4", node1, node1, node1);
    }

    // 有一个为空
    public static void test5(){
        ListNode node1 = ListUtils.createListNode(1);
        ListNode node2 = ListUtils.createListNode(2);
        ListNode node3 = ListUtils.createListNode(3);
        ListNode node4 = ListUtils.createListNode(4);
        ListNode node5 = ListUtils.createListNode(5);


        ListUtils.connectListNode(node1, node2);
        ListUtils.connectListNode(node2, node3);
        ListUtils.connectListNode(node3, node4);
        ListUtils.connectListNode(node4, node5);

//        ListNodeUtils.printList(node1);
        test("test5", node1, null, null);
    }

    //都为空
    public static void test6(){
        test("test6", null, null, null);
    }

    /**
     * 使用栈
     */
//    public static ListNode findFirstCommonNode(ListNode head1, ListNode head2){
//        if(head1 == null || head2 == null)
//            return null;
//        ListNode pHead1 = head1, pHead2 = head2;
//        Stack<ListNode> stack1 = new Stack<>();
//        Stack<ListNode> stack2 = new Stack<>();
//        while(pHead1 != null && pHead2 != null){
//            stack1.push(pHead1);
//            stack2.push(pHead2);
//            pHead1 = pHead1.next;
//            pHead2 = pHead2.next;
//        }
//        while(pHead1 != null){
//            stack1.push(pHead1);
//            pHead1 = pHead1.next;
//        }
//        while(pHead2 != null){
//            stack2.push(pHead2);
//            pHead2 = pHead2.next;
//        }
//        ListNode commonNode = null;
//        while(!stack1.isEmpty() && !stack2.isEmpty()){
//            if(stack1.peek() == stack2.peek()){
//                commonNode = stack1.pop();
//                stack2.pop();
//            }
//            else
//                return commonNode;
//        }
//        //两个栈都空，说明两个链表完全重合
//        if(stack1.isEmpty() && stack2.isEmpty())
//            return commonNode;
//        return null;
//    }

    /**
     * 也是两次遍历
     */
//    public static ListNode findFirstCommonNode(ListNode head1, ListNode head2){
//        if(head1 == null || head2 == null)
//            return null;
//        int length1 = getListLength(head1);
//        int length2 = getListLength(head2);
//        int lengthDif = length1 > length2 ? (length1 - length2) : (length2 - length1);
//        ListNode longList = null, shortList = null;
//        if(length1 > length2){
//            longList = head1;
//            shortList = head2;
//        }
//        else {
//            longList = head2;
//            shortList = head1;
//        }
//
//        for(int i  = 0; i < lengthDif; i++)
//            longList = longList.next;
//
//        while(longList != null && shortList != null && longList != shortList){
//            longList = longList.next;
//            shortList = shortList.next;
//        }
//        ListNode commondNode = longList;
//
//        return commondNode;
//    }
//
//    private static int getListLength(ListNode head){
//        if(head == null)
//            return -1;
//        ListNode pHead = head;
//        int length = 0;
//        while(pHead != null){
//            length++;
//            pHead = pHead.next;
//        }
//        return length;
//    }

    /**
     * 简洁算法
     */
    public static ListNode findFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        ListNode l1 = pHead1, l2 = pHead2;
        while (l1 != l2) {
            l1 = (l1 == null) ? pHead2 : l1.next;
            l2 = (l2 == null) ? pHead1 : l2.next;
        }
        return l1;
    }
}
