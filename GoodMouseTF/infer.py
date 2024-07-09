


while True:
    
    ret, frame=cap.read()
    if not ret:
        continue
    if frame is None:
        continue
    if camid==1:
        frame=cv2.flip(frame,1)
        
    img = preprocess_image_for_tflite32(frame,128)

    
    if bi_hand==False:
        # 设置输入数据
        result = fast_interpreter.set_input_tensor(0, img.data)
        if result != 0:
            print("interpreter set_input_tensor() failed")
        # 执行推理
        result = fast_interpreter.invoke()
        if result != 0:
            print("interpreter set_input_tensor() failed")
        # 获取输出数据
        raw_boxes = fast_interpreter.get_output_tensor(0)
        if raw_boxes is None:
            print("sample : interpreter->get_output_tensor() 1 failed !")

        classificators = fast_interpreter.get_output_tensor(1)
        if classificators is None:
            print("sample : interpreter->get_output_tensor() 0 failed !")
    
        detections = blazeface(raw_boxes, classificators, anchors)

        x_min,y_min,x_max,y_max=plot_detections(frame, detections[0])
        
        if len(detections[0])>0 :
            bi_hand=True
    if bi_hand:
        hand_nums=len(detections[0])
        if hand_nums>2:
            hand_nums=2
        for i in range(hand_nums):
            
            print(x_min,y_min,x_max,y_max)
            xmin=max(0,x_min[i])
            ymin=max(0,y_min[i])
            xmax=min(frame.shape[1],x_max[i])
            ymax=min(frame.shape[0],y_max[i])
    
            roi_ori=frame[ymin:ymax, xmin:xmax]
            roi =preprocess_image_for_tflite32(roi_ori,224)
               
            # 设置输入数据
            result = fast_interpreter1.set_input_tensor(0, roi.data)
            if result != 0:
                print("interpreter set_input_tensor() failed")
            # 执行推理
            result = fast_interpreter1.invoke()
            if result != 0:
                print("interpreter set_input_tensor() failed")
            # 获取输出数据
            mesh = fast_interpreter1.get_output_tensor(0)
            if mesh is None:
                print("sample : interpreter->get_output_tensor() 1 failed !")

            bi_hand=False
            mesh = mesh.reshape(21, 3)/224
            cx, cy = calc_palm_moment(roi_ori, mesh)
            draw_landmarks(roi_ori,cx,cy,mesh)
            frame[ymin:ymax, xmin:xmax]=roi_ori
    
    
    cv2.imshow("", frame)