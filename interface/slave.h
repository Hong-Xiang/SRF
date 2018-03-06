class Operation;
class Node
{
    size_t host_id;
    bool is_ready;
    vector<Node> dependency;
    vector<Operation> ops;

};

class Slave
{
size_t host_id;
  public:
    Slave();

    vector<Node> nodes;
    
    vector<(vector<Node>, Operation, Node)> to_calculate;

<NodeInOtherHost, RPC_READ, NodeInThisHost>
    void run()
    {

        while (true)
        {
            inputs, op, out = to_calculate.peep()
            if out.is_ready {
                to_calculate.pop();
            } else {
                if all(inputs.is_ready):
                    node.operation(inputs)
                    out.is_ready = true;
                else {
                    to_calculate.push(inputs.filter(lambda x: !x.is_ready))
                }

            }

        }
    }
};
