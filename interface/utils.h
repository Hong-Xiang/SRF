class Vec3;

class MemeoryManager
{
    vector<bool> occupied;
    float *pool;
    // if (occupied[i]) pool[i*BLOCK_SIZE, (i+1)*BLOCK_SIZE]

    static public : bind(size_t size)
    {
        return new float[size];
        for (int i = 0; i < this.occupied.length(); ++i)
        {
            if (!this.occupied[i])
            {
                this.occupied[i] = true;
                return pool + i * BLOCK_SIZE;
            }
        }

        this.pool[this.current_pos] = false result = this.current_pos this.current_pos += size;
        return result
    }

    static public : release(this.data)
    {
        this.occupied[(this.data - this.pool) / BLOCK_SIZE] = false;
        delete[] data;
    }
}

class Tensor
{
    size_t tid;
    float *data;

  public:
    float *data_ptr()
    {
        if (this.data == nullptr)
        {
            this.data = MemeoryManager::bind(size_t size);
        }
        return this.data;
    }

    void release()
    {
        MemeoryManager::release(this.data);
    }
};