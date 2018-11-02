
# -*- coding: utf-8 -*-


def display_image():
    """
    display heatmap & origin image
    :return:
    """
    from dataset_prepare import CocoMetadata, CocoPose
    from pycocotools.coco import COCO
    from os.path import join
    from dataset import _parse_function

    BASE_PATH = "~/data/ai_challenger"

    import os
    # os.chdir("..")

    ANNO = COCO(
        join(BASE_PATH, "ai_challenger_valid.json")
    )
    train_imgIds = ANNO.getImgIds()

    img, heat = _parse_function(train_imgIds[100], ANNO)

    CocoPose.display_image(img, heat, pred_heat=heat, as_numpy=False)

    from PIL import Image
    for _ in range(heat.shape[2]):
        data = CocoPose.display_image(img, heat, pred_heat=heat[:, :, _:(_ + 1)], as_numpy=True)
        im = Image.fromarray(data)
        im.save("test_heatmap/heat_%d.jpg" % _)


def saved_model_graph():
    """
    save the graph of model and check it in tensorboard
    :return:
    """

    from os.path import join
    from network_mv2_cpm_2 import build_network
    import tensorflow as tf
    import os

    INPUT_WIDTH = 256
    INPUT_HEIGHT = 256

    input_node = tf.placeholder(tf.float32, shape=(1, INPUT_WIDTH, INPUT_HEIGHT, 3),
                                name='image')
    build_network(input_node, False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(
            join("tensorboard/test_graph/"),
            sess.graph
        )
        sess.run(tf.global_variables_initializer())


def metric_prefix(input_width, input_height):
    """
    output the calculation of you model
    :param input_width:
    :param input_height:
    :return:
    """
    import tensorflow as tf
    from networks import get_network
    import os

    input_node = tf.placeholder(tf.float32, shape=(1, input_width, input_height, 3),
                                name='image')
    get_network("mv2_cpm_2", input_node, False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    run_meta = tf.RunMetadata()
    with tf.Session(config=config) as sess:
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        print("opts {:,} --- paras {:,}".format(flops.total_float_ops, params.total_parameters))
        sess.run(tf.global_variables_initializer())


def run_with_frozen_pb(img_path, input_w_h, frozen_graph, output_node_names):
    import tensorflow as tf
    import cv2
    import numpy as np
    import os
    from dataset_prepare import CocoPose
    with tf.gfile.GFile(frozen_graph, "rb") as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name=""
    )

    graph = tf.get_default_graph()
    image = graph.get_tensor_by_name("image:0")
    output = graph.get_tensor_by_name("%s:0" % output_node_names)
    
    cap = cv2.VideoCapture(0)
    i = 0
    with tf.Session() as sess:
      while True:
        
        ret, image_0 = cap.read()
        #cv2.imshow("input", image_0)

        key = cv2.waitKey(10)
        if key == 27:
          break
        #image_0 = cv2.imread(img_path)
        h, w, _ = image_0.shape
        image_ = cv2.resize(image_0, (input_w_h, input_w_h), interpolation=cv2.INTER_AREA)

        import time
        start = time.time()
        heatmaps = sess.run(output, feed_dict={image: [image_]})
        print(time.time()-start)
        print(heatmaps.shape)
        from PIL import Image
        # for heatmap
        data = np.sum(heatmaps[0,:,:,:],axis=2)
        #print(data.shape)
        im = Image.fromarray((data*255).astype(np.uint8),'L')
        im.save("test/heat_%d.jpg" % i)
        # for keypoints
        m_num,_,_ = heatmaps[0,:,:,:].shape 
        data = heatmaps[0,:,:,:].reshape(m_num*m_num,14).T
        print(np.max(data,axis=1))
        
        # may not be strict threshold
        lower_threshold_indices = data < 0.15
        data[lower_threshold_indices] = 0
        data = np.argmax(data,axis=1)
      
        Y = data//m_num*(192//m_num)
        X = np.mod(data,m_num)*(192//m_num)
        for k in range(14):
           #print(X[k],Y[k])
           cv2.circle(image_,(X[k], Y[k]), 1, (0,255,0),-1)
        image_p = cv2.resize(image_, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imshow("input", image_p)
        cv2.imwrite("test/point_%d.jpg"%i, image_p)
        i = i+1


if __name__ == '__main__':
    run_with_frozen_pb(
         "mo1.jpg",
         192,
         "./model_hourglass.pb",
         "hourglass_out_3"
     )

