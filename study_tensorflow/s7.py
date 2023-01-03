
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost=0
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            avg_cost+=c/total_batch
        if (epoch+1)%display_step==0:
            print('epoch:','%04d'%(epoch+1),'cost=','{:.9f}'.format(avg_cost))
    print('finished!')

    save_path=r'./checkpoints/0101.ckpt'
    save_path=saver.save(sess,save_path)