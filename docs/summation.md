# 可扩展重建框架(SRF)第一阶段开发总结

## 开发结果

1. 重建框架本身的底层初步开发完成，可以将重建的计算图分布到多节点多GPU上；
2. 计算图最底层算符开发已经处于debug阶段；
3. 重建框架上层API（封装和调用）预研完成，封装到容器完成，简易的调用尚需开发；
4. 重建框架分布式部分仍需优化，以降低不需要的内存占用。

## 经验教训

本次开发第一阶段任务目标未能达成，没能在截稿日前提交出结果，我总结有一下原因：

### 客观原因

1. 开发时间相对比较短，从三月六日创建项目到四月二日的截止日期，开发时间仅一个月。
   如果仅仅是开发框架时间应该是足够的，然而实际上后期更多时间都是在debug重建程序，
   提高分布式易用性等问题，最终没能全部完成。

2. 硬件支持出现意外，GPU服务器迟迟没有到，这也是导致第一点中部分额外工作的原因。

### 主观原因

1. 构架上不够完善。第一版的框架控制节点和计算节点耦合性较强，给后来拓展多节点支持带来了困难；
2. 测试不够重视，测试用例的整理没有提到最高优先级，导致后期debug时间消耗巨大；
3. 项目管理不合理，任务没有充分细分，没有后备方案，也没有充分沟通；在出现时间不够的情况下，
   没有及时加入人力；
4. 基础设施管理不规范，在开发程序的过程中，集群蒙特卡罗占用比较高，提高了一定的开发成本；
