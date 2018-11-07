package smc;
import java.lang.IllegalArgumentException;


public class Goal {

    private String target;

    private String operator;

    private float tresshold;

    public Goal (String target, String operator, float tresshold) throws IllegalArgumentException
    {
        this.target = target;

        if (operator != "<" && operator != ">" && operator != "<=" && operator != ">=" && operator != "==" && operator != "!=")
        {
            throw new IllegalArgumentException("The operator can only be < or > or <= or >= or == or != .");
        }
        this.operator = operator;
        this.tresshold = tresshold;
    }

    public String getTarget()
    {
        return this.target;
    }

    public String getOperator()
    {
        return this.operator;
    }

    public float getTresshold()
    {
        return this.tresshold;
    }

    public boolean evaluate(float value) throws IllegalArgumentException
    {
        String op = getOperator();
        float tress = getTresshold();

        if(op == "<")
        {
            return value < tress;
        }
        else if(op == ">")
        {
            return value > tress;
        }
        else if(op == "<="){
            return value <= tress;
        }
        else if(op == ">=")
        {
            return value >= tress;
        }
        else if(op == "==")
        {
            return value == tress;
        }
        else if(op == "!=")
        {
            return value != tress;
        }
        else
        {
            throw new IllegalArgumentException("Illegal operator.");
        }
    }

}