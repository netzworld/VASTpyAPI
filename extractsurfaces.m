function [] = extractsurfaces()
  global vdata;
  
  if (~checkconnection()) return; end
  
  if (vdata.data.exportobj.disablenetwarnings==1)
    vdata.vast.seterrorpopupsenabled(74,0); %network connection error
    vdata.vast.seterrorpopupsenabled(75,0); %unexpected data
    vdata.vast.seterrorpopupsenabled(76,0); %unexpected length
  end
  
  set(vdata.ui.cancelbutton,'Enable','on');
  set(vdata.ui.message,'String',{'Exporting Surfaces ...','Loading Metadata ...'});
  pause(0.1);
  
  param=vdata.data.exportobj;
  rparam=vdata.data.region;
  extractseg=1;
  %if ((param.extractwhich==5)||(param.extractwhich==6)||(param.extractwhich==7)||(param.extractwhich==8)||(param.extractwhich==9)) extractseg=0; end
  if ((param.extractwhich>=5)&&(param.extractwhich<=10)) extractseg=0; end
    
  if (extractseg)
    [data,res] = vdata.vast.getallsegmentdatamatrix();
    [name,res] = vdata.vast.getallsegmentnames();
    seglayername=getselectedseglayername();
    name(1)=[]; %remove 'Background'
    maxobjectnumber=max(data(:,1));
    
    [selectedlayernr, selectedemlayernr, selectedseglayernr, res] = vdata.vast.getselectedlayernr();
    mipdatasize=vdata.vast.getdatasizeatmip(param.miplevel,selectedseglayernr);
    
    %if (param.erodedilate==1) param.overlap=2; end %pad one pixel more so that erodedilate works correctly (hopefully); crop later
  else
    switch param.extractwhich
      case 5 %RGB 50%
        name={'Red Layer', 'Green Layer', 'Blue Layer'};
      case 6 %Brightness 50%
        param.lev=128;
        name={'Brightness 128'};
      case 7 %16 levels
        param.lev=8:16:256;
        for i=1:length(param.lev)
          name{i}=sprintf('B%03d',param.lev(i));
        end
      case 8 %32 levels
        param.lev=4:8:256;
        for i=1:length(param.lev)
          name{i}=sprintf('B%03d',param.lev(i));
        end
      case 9 %64 levels
        param.lev=2:4:256;
        for i=1:length(param.lev)
          name{i}=sprintf('B%03d',param.lev(i));
        end
      %case 10 %Up to 2^24 objects
        %colorcounts=zeros(256*256*256,1,'uint32'); %64 MB
    end
  end
  

  
  [selectedlayernr, selectedemlayernr, selectedsegmentlayernr, res]=vdata.vast.getselectedlayernr();
  zscale=1;
  zmin=rparam.zmin;
  zmax=rparam.zmax;
  if (param.miplevel>0)
    if (extractseg)
      [mipscalematrix, res] = vdata.vast.getmipmapscalefactors(selectedsegmentlayernr);
    else
      [mipscalematrix, res] = vdata.vast.getmipmapscalefactors(selectedemlayernr);
    end
    if (param.miplevel>0)
      zscale=mipscalematrix(param.miplevel,3);
    else
      zscale=1;
    end
    if (zscale~=1)
      zmin=floor(zmin/zscale);
      zmax=floor(zmax/zscale);
    end
  end
  
  xmin=bitshift(rparam.xmin,-param.miplevel);
  xmax=bitshift(rparam.xmax,-param.miplevel)-1;
  ymin=bitshift(rparam.ymin,-param.miplevel);
  ymax=bitshift(rparam.ymax,-param.miplevel)-1;
  
  %mipfact=bitshift(1,param.miplevel);
  if (param.miplevel>0)
    mipfactx=mipscalematrix(param.miplevel,1); 
    mipfacty=mipscalematrix(param.miplevel,2); 
    mipfactz=mipscalematrix(param.miplevel,3);
  else
    mipfactx=1; 
    mipfacty=1; 
    mipfactz=1;
  end
  
  if (((xmin==xmax)||(ymin==ymax)||(zmin==zmax))&&(param.closesurfaces==0))
    waitfor(warndlg('ERROR: The Matlab surface script needs a volume which is at least two pixels wide in each direction to work. Please adjust "Render from area" values, or enable "Close surface sides".','VastTools OBJ Exporting'));
    set(vdata.ui.message,'String','Canceled.');
    set(vdata.ui.cancelbutton,'Enable','off');
    vdata.state.lastcancel=0;
    return;
  end
  
  if (extractseg)
    % Compute full name (including folder names) from name and hierarchy
    if (param.includefoldernames==1)
      fullname=name;
      for i=1:1:size(data,1)
        j=i;
        while data(j,14)~=0 %Check if parent is not 0
          j=data(j,14);
          fullname{i}=[name{j} '.' fullname{i}];
        end
      end
      name=fullname;
    end
  
    % Compute list of objects to export
    switch param.extractwhich
      case 1  %All segments individually, uncollapsed
        objects=uint32([data(:,1) data(:,2)]);
        vdata.vast.setsegtranslation([],[]);
        
      case 2  %All segments, collapsed as in Vast
        %4: Collapse segments as in the view during segment text file exporting
        objects=unique(data(:,18));
        objects=uint32([objects data(objects,2)]);
        vdata.vast.setsegtranslation(data(:,1),data(:,18));
        
      case 3  %Selected segment and children, uncollapsed
        selected=find(bitand(data(:,2),65536)>0);
        if (min(size(selected))==0)
          objects=uint32([data(:,1) data(:,2)]);
        else
          selected=[selected getchildtreeids_seg(data,selected)];
          objects=uint32([selected' data(selected,2)]);
        end
        vdata.vast.setsegtranslation(data(selected,1),data(selected,1));
        
      case 4  %Selected segment and children, collapsed as in VAST
        selected=find(bitand(data(:,2),65536)>0);
        if (min(size(selected))==0)
          %None selected: choose all, collapsed
          selected=data(:,1);
          objects=unique(data(:,18));
        else
          selected=[selected getchildtreeids_seg(data,selected)];
          objects=unique(data(selected,18));
        end
        
        objects=uint32([objects data(objects,2)]);
        vdata.vast.setsegtranslation(data(selected,1),data(selected,18));
    end
  end
  
  
  % Compute number of blocks in volume
  nrxtiles=0; tilex1=xmin;
  while (tilex1<=xmax)
    tilex1=tilex1+param.blocksizex-param.overlap;
    nrxtiles=nrxtiles+1;
  end
  nrytiles=0; tiley1=ymin;
  while (tiley1<=ymax)
    tiley1=tiley1+param.blocksizey-param.overlap;
    nrytiles=nrytiles+1;
  end
  nrztiles=0; tilez1=zmin;
  if (param.slicestep==1)
    slicenumbers=zmin:zmax;
    while (tilez1<=zmax)
      tilez1=tilez1+param.blocksizez-param.overlap;
      nrztiles=nrztiles+1;
    end
  else
    slicenumbers=zmin:vdata.data.exportobj.slicestep:zmax;
    nrztiles=ceil(size(slicenumbers,2)/(param.blocksizez-param.overlap));
    j=1;
    for p=1:param.blocksizez-param.overlap:size(slicenumbers,2)
      pe=min([p+param.blocksizez-1 size(slicenumbers,2)]);
      blockslicenumbers{j}=slicenumbers(p:pe);
      j=j+1;
    end
  end
  param.nrxtiles=nrxtiles; param.nrytiles=nrytiles; param.nrztiles=nrztiles;
  
  
  if (param.usemipregionconstraint)   
    %Calculate export region at regionconstraint mip
    cxmin=bitshift(rparam.xmin,-param.mipregionmip);
    cxmax=bitshift(rparam.xmax,-param.mipregionmip)-1;
    cymin=bitshift(rparam.ymin,-param.mipregionmip);
    cymax=bitshift(rparam.ymax,-param.mipregionmip)-1;
    czmin=rparam.zmin;
    czmax=rparam.zmax;
    if (param.mipregionmip>0)
      czscale=mipscalematrix(param.mipregionmip,3);
      czmin=floor(czmin/czscale);
      czmax=floor(czmax/czscale);
    end
    
    if (extractseg)
      %Load complete region of segmentation source layer at constraint mip level (vdata.data.exportobj.mipregionmip equals param.mipregionmip)
      message={'Exporting Surfaces ...',sprintf('Loading segmentation at mip %d for mip constraint...',param.mipregionmip)};
      set(vdata.ui.message,'String',message);
      pause(0.01);
      if (param.slicestep==1)
        [mcsegimage,values,numbers,bboxes,res] = vdata.vast.getsegimageRLEdecodedbboxes(param.mipregionmip,cxmin,cxmax,cymin,cymax,czmin,czmax,0); %order of dimensions in mcsegimage:(x,y,z)
      else
        %ps=param.slicestep*mipscalematrix(param.mipregionmip,3)/mipscalematrix(param.miplevel,3);
        s=czmin:param.slicestep:czmax;
        mcsegimage=zeros(cxmax-cxmin+1,cymax-cymin+1,length(s));
        for slice=czmin:param.slicestep:czmax
          [mcsegslice,values,numbers,bboxes,res] = vdata.vast.getsegimageRLEdecodedbboxes(param.mipregionmip,cxmin,cxmax,cymin,cymax,slice,slice,0);
          mcsegimage(:,:,slice)=mcsegslice;
        end
      end
    else
      %Load complete region of segmentation source layer at constraint mip level (vdata.data.exportobj.mipregionmip equals param.mipregionmip)
      message={'Exporting Surfaces ...',sprintf('Loading screenshots at mip %d for mip constraint...',param.mipregionmip)};
      set(vdata.ui.message,'String',message);
      pause(0.01);
      %[mcsegimage,values,numbers,bboxes,res] = vdata.vast.getsegimageRLEdecodedbboxes(param.mipregionmip,cxmin,cxmax,cymin,cymax,czmin,czmax,0); %order of dimensions in mcsegimage:(x,y,z)
      if (param.slicestep==1)
        [mcsegimage,res] = vdata.vast.getscreenshotimage(param.mipregionmip,cxmin,cxmax,cymin,cymax,czmin,czmax, 0); %order of dimensions: y,x,z,c
        mcsegimage=permute(sum(mcsegimage,4)>0,[2 1 3]);
      else
        s=czmin:param.slicestep:czmax;
        mcsegimage=zeros(cxmax-cxmin+1,cymax-cymin+1,length(s));
        for slice=czmin:param.slicestep:czmax
          [mcsegslice,res] = vdata.vast.getscreenshotimage(param.mipregionmip,cxmin,cxmax,cymin,cymax,slice,slice, 0); %order of dimensions: y,x,z,c
          mcsegimage(:,:,slice)=mcsegslice;
        end
        mcsegimage=permute(sum(mcsegimage,4)>0,[2 1 3]);
      end
    end
    
    %Translate to 0/1 mask depending on what objects are exported
    message={'Exporting Surfaces ...',sprintf('Processing image at mip %d for mip constraint...',param.mipregionmip)};
    set(vdata.ui.message,'String',message);
    pause(0.01);
    mcsegimage=(mcsegimage>0); %Here I am assuming that VAST pre-translates the segment data so that all nonzero pixels belong to pbjects which are going to be exported
      
    %Dilate mask by region padding
    sz=param.mipregionpadding*2+1;
    strel=ones(sz,sz,sz);
    mcsegimage=imdilate(mcsegimage,strel);
    %Generate 3D matrix of block loading flags
    mc_loadflags=zeros(nrxtiles,nrytiles,nrztiles);
    
    cmipfactx=mipscalematrix(param.mipregionmip,1)/mipscalematrix(param.miplevel,1);
    cmipfacty=mipscalematrix(param.mipregionmip,2)/mipscalematrix(param.miplevel,2);
    cmipfactz=mipscalematrix(param.mipregionmip,3)/mipscalematrix(param.miplevel,3);
    
    tilez1=zmin;
    for (tz=1:nrztiles)
      tilez2=tilez1+param.blocksizez-1;
      if (tilez2>zmax) tilez2=zmax; end
      tiley1=ymin;
      for (ty=1:nrytiles)
        tiley2=tiley1+param.blocksizey-1;
        if (tiley2>ymax) tiley2=ymax; end
        tilex1=xmin;
        for (tx=1:nrxtiles)
          tilex2=tilex1+param.blocksizex-1;
          if (tilex2>xmax) tilex2=xmax; end
          
          %Compute tile coords on constraint mip
          cminx=max([1 floor((tilex1-xmin)/cmipfactx)+1]);
          cmaxx=min([size(mcsegimage,1) ceil((tilex2-xmin)/cmipfactx)+1]);
          cminy=max([1 floor((tiley1-ymin)/cmipfacty)+1]);
          cmaxy=min([size(mcsegimage,2) ceil((tiley2-ymin)/cmipfacty)+1]);
          cminz=max([1 floor((tilez1-zmin)/cmipfactz)+1]);
          cmaxz=min([size(mcsegimage,3) ceil((tilez2-zmin)/cmipfactz)+1]);
          
          %Crop out region from constraint mip
          cropregion=mcsegimage(cminx:cmaxx,cminy:cmaxy,cminz:cmaxz); %order of dimensions in mcsegimage:(x,y,z)
          mc_loadflags(tx,ty,tz)=max(cropregion(:)); %This flag will be 1 if any of the voxels in cropregion are 1
          
          tilex1=tilex1+param.blocksizex-param.overlap;
        end
        tiley1=tiley1+param.blocksizey-param.overlap;
      end
      tilez1=tilez1+param.blocksizez-param.overlap;
    end

  end
  
  
  if (extractseg)
    %Go through all blocks and extract surfaces
    param.farray=cell(maxobjectnumber,param.nrxtiles,param.nrytiles,param.nrztiles);
    param.varray=cell(maxobjectnumber,param.nrxtiles,param.nrytiles,param.nrztiles);
    param.objects=objects;
    param.objectvolume=zeros(size(objects,1),1);
  else
    if (param.extractwhich==10)
      %Up to 2^24 objects
      %param.farray=cell(256*256*256,param.nrxtiles,param.nrytiles,param.nrztiles);
      %param.varray=cell(256*256*256,param.nrxtiles,param.nrytiles,param.nrztiles);
      param.fvindex=sparse(256*256*256,(param.nrxtiles+1)*(param.nrytiles+1)*(param.nrztiles+1));
      %param.vindex=sparse(256*256*256,param.nrxtiles*param.nrytiles*param.nrztiles);
      param.objectvolume=zeros(256*256*256,1); %64 MB
    else
      param.farray=cell(3,param.nrxtiles,param.nrytiles,param.nrztiles);
      param.varray=cell(3,param.nrxtiles,param.nrytiles,param.nrztiles);
      param.objects=[(1:length(name))' zeros(length(name),1)];
      param.objectvolume=zeros(length(name),1);
    end
  end
  
  tilez1=zmin; tz=1; blocknr=1;
  while ((tz<=nrztiles)&&(vdata.state.lastcancel==0))
    tilez2=tilez1+param.blocksizez-1;
    if (tilez2>zmax) tilez2=zmax; end
    tilezs=tilez2-tilez1+1;
    tiley1=ymin; ty=1;
    while ((ty<=nrytiles)&&(vdata.state.lastcancel==0))
      tiley2=tiley1+param.blocksizey-1;
      if (tiley2>ymax) tiley2=ymax; end
      tileys=tiley2-tiley1+1;
      tilex1=xmin; tx=1;
      while ((tx<=nrxtiles)&&(vdata.state.lastcancel==0))
        tilex2=tilex1+param.blocksizex-1;
        if (tilex2>xmax) tilex2=xmax; end
        tilexs=tilex2-tilex1+1;
        
        if (extractseg)
          if ((param.usemipregionconstraint==0)||(mc_loadflags(tx,ty,tz)==1))
            message={'Exporting Surfaces ...',sprintf('Loading Segmentation Cube (%d,%d,%d) of (%d,%d,%d)...',tx,ty,tz,nrxtiles,nrytiles,nrztiles)};
            set(vdata.ui.message,'String',message);
            pause(0.01);
            %Read this cube
            
            if (param.erodedilate==1)
              edx=ones(3,2);
              if (tilex1==0) edx(1,1)=0; end
              if (tilex2>=mipdatasize(1)) edx(1,2)=0; end
              if (tiley1==0) edx(2,1)=0; end
              if (tiley2>=mipdatasize(2)) edx(2,2)=0; end
              if (tilez1==0) edx(3,1)=0; end
              if (tilez2>=mipdatasize(3)) edx(3,2)=0; end
            else
              edx=zeros(3,2);
            end
            
            
            if (vdata.data.exportobj.slicestep==1)
              [segimage,values,numbers,bboxes,res] = vdata.vast.getsegimageRLEdecodedbboxes(param.miplevel,tilex1-edx(1,1),tilex2+edx(1,2),tiley1-edx(2,1),tiley2+edx(2,2),tilez1-edx(3,1),tilez2+edx(3,2),0);
            else
              bs=blockslicenumbers{tz};
              if (edx(3,1)==1)
                bs=[bs(1)-param.slicestep bs];
              end
              if (edx(3,2)==1)
                bs=[bs bs(end)+param.slicestep];
              end
              %segimage=uint16(zeros(tilex2-tilex1+1,tiley2-tiley1+1,size(bs,2)));
              segimage=uint16(zeros(tilex2-tilex1+1+edx(1,1)+edx(1,2),tiley2-tiley1+1+edx(2,1)+edx(2,2),size(bs,2)));
              numarr=int32(zeros(maxobjectnumber,1));
              bboxarr=zeros(maxobjectnumber,6)-1;
              firstblockslice=bs(1);
              for i=1:1:size(bs,2)
                [ssegimage,svalues,snumbers,sbboxes,res] = vdata.vast.getsegimageRLEdecodedbboxes(param.miplevel,tilex1-edx(1,1),tilex2+edx(1,2),tiley1-edx(2,1),tiley2+edx(2,2),bs(i),bs(i),0);
                segimage(:,:,i)=ssegimage;
                snumbers(svalues==0)=[];
                sbboxes(svalues==0,:)=[];
                sbboxes(:,[3 6])=sbboxes(:,[3 6])+i-1;
                svalues(svalues==0)=[];
                if (min(size(svalues))>0)
                  numarr(svalues)=numarr(svalues)+snumbers;
                  bboxarr(svalues,:)=vdata.vast.expandboundingboxes(bboxarr(svalues,:),sbboxes);
                end
              end
              values=find(numarr>0);
              numbers=numarr(values);
              bboxes=bboxarr(values,:);
            end
            
            if (param.erodedilate==1)
              strel=ones(2,2,2);
              segimage=imopen(segimage,strel); %This is supposed to erode by 1 voxel and then dilate by one voxel, removing 1-voxel-thin objects
              %Crop padding
              %segimage=segimage(2:end-1,2:end-1,2:end-1);
              segimage=segimage(1+edx(1,1):end-edx(1,2),1+edx(2,1):end-edx(2,2),1+edx(3,1):end-edx(3,2));
              
              if (min(size(bboxes))>0)
                %adjust bboxes
                if (edx(1,1)>0)
                  bb=bboxes(:,[1 4])-1;
                  bb(bb==0)=1;
                  bboxes(:,[1 4])=bb;
                end
                if (edx(1,2)>0)
                  bb=bboxes(:,[1 4]);
                  bb(bb>size(segimage,1))=size(segimage,1);
                  bboxes(:,[1 4])=bb;
                end
                
                if (edx(2,1)>0)
                  bb=bboxes(:,[2 5])-1;
                  bb(bb==0)=1;
                  bboxes(:,[2 5])=bb;
                end
                if (edx(2,2)>0)
                  bb=bboxes(:,[2 5]);
                  bb(bb>size(segimage,2))=size(segimage,2);
                  bboxes(:,[2 5])=bb;
                end
                
                if (edx(3,1)>0)
                  bb=bboxes(:,[3 6])-1;
                  bb(bb==0)=1;
                  bboxes(:,[3 6])=bb;
                end
                if (edx(3,2)>0)
                  bb=bboxes(:,[3 6]);
                  bb(bb>size(segimage,3))=size(segimage,3);
                  bboxes(:,[3 6])=bb;
                end
              end
              
            end
          end
          
          
        else
          if ((param.usemipregionconstraint==0)||(mc_loadflags(tx,ty,tz)==1))
            message={'Exporting Surfaces ...',sprintf('Loading Screenshot Cube (%d,%d,%d) of (%d,%d,%d)...',tx,ty,tz,nrxtiles,nrytiles,nrztiles)};
            set(vdata.ui.message,'String',message);
            pause(0.01);
            %Read this cube
            if (vdata.data.exportobj.slicestep==1)
              [scsimage,res] = vdata.vast.getscreenshotimage(param.miplevel,tilex1,tilex2,tiley1,tiley2,tilez1,tilez2,1,1);
              if (tilez1==tilez2)
                %in this case a 3d array will be returned, but a 4d array with singular dimension 3 is expected below.
                scsimage=reshape(scsimage,size(scsimage,1),size(scsimage,2),1,size(scsimage,3));
              end
            else
              bs=blockslicenumbers{tz};
              %scsimage=uint8(zeros(tilex2-tilex1+1,tiley2-tiley1+1,size(bs,2),3));
              scsimage=uint8(zeros(tiley2-tiley1+1,tilex2-tilex1+1,size(bs,2),3));
              firstblockslice=bs(1);
              for i=1:1:size(bs,2)
                [scsslice,res] = vdata.vast.getscreenshotimage(param.miplevel,tilex1,tilex2,tiley1,tiley2,bs(i),bs(i),1,1);
                scsimage(:,:,i,:)=scsslice;
              end
            end
          end
        end
        
        if (extractseg)
          if ((param.usemipregionconstraint==0)||(mc_loadflags(tx,ty,tz)==1))
            message={'Exporting Surfaces ...',sprintf('Processing Segmentation Cube (%d,%d,%d) of (%d,%d,%d)...',tx,ty,tz,nrxtiles,nrytiles,nrztiles)};
            set(vdata.ui.message,'String',message);
            pause(0.01);
            
            numbers(values==0)=[];
            bboxes(values==0,:)=[];
            values(values==0)=[];
            
            if (min(size(values))>0)
              % VAST now translates the voxel data before transmission because Matlab is too slow.
              
              xvofs=0; yvofs=0; zvofs=0; ttxs=tilexs; ttys=tileys; ttzs=tilezs;
              
              
              
              %Close surfaces
              if (vdata.data.exportobj.closesurfaces==1)
                if (tx==1)
                  segimage=cat(1,zeros(1,size(segimage,2),size(segimage,3)),segimage);
                  bboxes(:,1)=bboxes(:,1)+1;
                  bboxes(:,4)=bboxes(:,4)+1;
                  xvofs=xvofs-1;
                  ttxs=ttxs+1;
                end
                if (ty==1)
                  segimage=cat(2,zeros(size(segimage,1),1,size(segimage,3)),segimage);
                  bboxes(:,2)=bboxes(:,2)+1;
                  bboxes(:,5)=bboxes(:,5)+1;
                  yvofs=yvofs-1;
                  ttys=ttys+1;
                end
                if (tz==1)
                  segimage=cat(3,zeros(size(segimage,1),size(segimage,2),1),segimage);
                  bboxes(:,3)=bboxes(:,3)+1;
                  bboxes(:,6)=bboxes(:,6)+1;
                  zvofs=zvofs-1;
                  ttzs=ttzs+1;
                end
                if (tx==nrxtiles)
                  segimage=cat(1,segimage,zeros(1,size(segimage,2),size(segimage,3)));
                  ttxs=ttxs+1;
                end
                if (ty==nrytiles)
                  segimage=cat(2,segimage,zeros(size(segimage,1),1,size(segimage,3)));
                  ttys=ttys+1;
                end
                if (tz==nrztiles)
                  segimage=cat(3,segimage,zeros(size(segimage,1),size(segimage,2),1));
                  ttzs=ttzs+1;
                end
              end
              
              %Extract all segments
              segnr=1;
              while ((segnr<=size(values,1))&&(vdata.state.lastcancel==0))
                seg=values(segnr);
                
                if (mod(segnr,10)==1)
                  set(vdata.ui.message,'String',[message sprintf('Objects %d-%d of %d ...',segnr,min([segnr+9 size(values,1)]),size(values,1))]);
                  pause(0.01);
                end
                
                bbx=bboxes(segnr,:);
                bbx=bbx+[-1 -1 -1 1 1 1];
                if (bbx(1)<1) bbx(1)=1; end
                if (bbx(2)<1) bbx(2)=1; end
                if (bbx(3)<1) bbx(3)=1; end
                if (bbx(4)>ttxs) bbx(4)=ttxs; end
                if (bbx(5)>ttys) bbx(5)=ttys; end
                if (bbx(6)>ttzs) bbx(6)=ttzs; end
                
                %Adjust extracted subvolumes to be at least 2 pixels in each direction
                if (bbx(1)==bbx(4))
                  if (bbx(1)>1)
                    bbx(1)=bbx(1)-1;
                  else
                    bbx(4)=bbx(4)+1;
                  end
                end
                if (bbx(2)==bbx(5))
                  if (bbx(2)>1)
                    bbx(2)=bbx(2)-1;
                  else
                    bbx(5)=bbx(5)+1;
                  end
                end
                if (bbx(3)==bbx(6))
                  if (bbx(3)>1)
                    bbx(3)=bbx(3)-1;
                  else
                    bbx(6)=bbx(6)+1;
                  end
                end
                
                subseg=segimage(bbx(1):bbx(4),bbx(2):bbx(5),bbx(3):bbx(6)); %(ymin:ymax,xmin:xmax,zmin:zmax);
                subseg=double(subseg==seg);
                
                if (min(size(subseg))<2)
                  subseg=subseg;
                end
                
                [f,v]=isosurface(subseg,0.5);
                if (size(v,1)>0)
                  %adjust coordinates for bbox and when we added empty slices at beginning
                  v(:,1)=v(:,1)+bbx(2)-1+yvofs;
                  v(:,2)=v(:,2)+bbx(1)-1+xvofs;
                  v(:,3)=v(:,3)+bbx(3)-1+zvofs;
                  
                  v(:,1)=v(:,1)+tiley1-1;
                  v(:,2)=v(:,2)+tilex1-1;
                  if (vdata.data.exportobj.slicestep==1)
                    v(:,3)=v(:,3)+tilez1-1;
                  else
                    v(:,3)=((v(:,3)-0.5)*vdata.data.exportobj.slicestep)+0.5+firstblockslice-1;
                  end
                  v(:,1)=v(:,1)*param.yscale*param.yunit*mipfacty;
                  v(:,2)=v(:,2)*param.xscale*param.xunit*mipfactx;
                  v(:,3)=v(:,3)*param.zscale*param.zunit*mipfactz;
                end
                param.farray{seg,tx,ty,tz}=f;
                param.varray{seg,tx,ty,tz}=v;
                
                segnr=segnr+1;
              end
            end
          end
        else
          if ((param.usemipregionconstraint==0)||(mc_loadflags(tx,ty,tz)==1))
            message={'Exporting Surfaces ...',sprintf('Processing Screenshot Cube (%d,%d,%d) of (%d,%d,%d)...',tx,ty,tz,nrxtiles,nrytiles,nrztiles)};
            set(vdata.ui.message,'String',message);
            pause(0.01);
            
            rcube=permute(squeeze(scsimage(:,:,:,1)),[2 1 3]);
            gcube=permute(squeeze(scsimage(:,:,:,2)),[2 1 3]);
            bcube=permute(squeeze(scsimage(:,:,:,3)),[2 1 3]);
            
            %Close surfaces
            xvofs=0; yvofs=0; zvofs=0; ttxs=tilexs; ttys=tileys; ttzs=tilezs;
            if (vdata.data.exportobj.closesurfaces==1)
              if (tx==1)
                rcube=cat(1,zeros(1,size(rcube,2),size(rcube,3)),rcube);
                gcube=cat(1,zeros(1,size(gcube,2),size(gcube,3)),gcube);
                bcube=cat(1,zeros(1,size(bcube,2),size(bcube,3)),bcube);
                xvofs=-1;
                ttxs=ttxs+1;
              end
              if (ty==1)
                rcube=cat(2,zeros(size(rcube,1),1,size(rcube,3)),rcube);
                gcube=cat(2,zeros(size(gcube,1),1,size(gcube,3)),gcube);
                bcube=cat(2,zeros(size(bcube,1),1,size(bcube,3)),bcube);
                yvofs=-1;
                ttys=ttys+1;
              end
              if (tz==1)
                rcube=cat(3,zeros(size(rcube,1),size(rcube,2),1),rcube);
                gcube=cat(3,zeros(size(gcube,1),size(gcube,2),1),gcube);
                bcube=cat(3,zeros(size(bcube,1),size(bcube,2),1),bcube);
                zvofs=-1;
                ttzs=ttzs+1;
              end
              if (tx==nrxtiles)
                rcube=cat(1,rcube,zeros(1,size(rcube,2),size(rcube,3)));
                gcube=cat(1,gcube,zeros(1,size(gcube,2),size(gcube,3)));
                bcube=cat(1,bcube,zeros(1,size(bcube,2),size(bcube,3)));
                ttxs=ttxs+1;
              end
              if (ty==nrytiles)
                rcube=cat(2,rcube,zeros(size(rcube,1),1,size(rcube,3)));
                gcube=cat(2,gcube,zeros(size(gcube,1),1,size(gcube,3)));
                bcube=cat(2,bcube,zeros(size(bcube,1),1,size(bcube,3)));
                ttys=ttys+1;
              end
              if (tz==nrztiles)
                rcube=cat(3,rcube,zeros(size(rcube,1),size(rcube,2),1));
                gcube=cat(3,gcube,zeros(size(gcube,1),size(gcube,2),1));
                bcube=cat(3,bcube,zeros(size(bcube,1),size(bcube,2),1));
                ttzs=ttzs+1;
              end
            end
            
            %Extract isosurfaces
            if (param.extractwhich==10)
              %Extract unique colors from screenshots as individual 3d objects
              colcube=double(bitshift(int32(rcube),16)+bitshift(int32(gcube),8)+int32(bcube));
              [num,val] = hist(colcube(:),unique(colcube(:))); %Alternative to count_unique
              num(val==0)=[];
              val(val==0)=[];
              param.objectvolume(val)=param.objectvolume(val)+num';
              
              if (min(size(val)>0))
                colnr=1;
               
                
                while ((colnr<=length(val))&&(vdata.state.lastcancel==0))
                  endcolnr=length(val); %min(colnr+15,length(val));
                  
                  pf=cell(endcolnr-colnr+1);
                  pv=cell(endcolnr-colnr+1);
                  parfor (pcolnr=1:endcolnr-colnr+1)
                    nr=colnr+pcolnr-1;
                    psubseg=double(colcube==val(nr));
                    %[pf{pcolnr-colnr+1},pv{pcolnr-colnr+1}]=isosurface(psubseg,0.5);
                    [lf,lv]=isosurface(psubseg,0.5);
                    pf{pcolnr}=lf;
                    pv{pcolnr}=lv;
                  end
                  
                  for pcolnr=colnr:endcolnr
                    v=pv{pcolnr-colnr+1};
                    f=pf{pcolnr-colnr+1};
                    if (size(v,1)>0)
                      %adjust coordinates for bbox and when we added empty slices at beginning
                      v(:,1)=v(:,1)+yvofs;
                      v(:,2)=v(:,2)+xvofs;
                      v(:,3)=v(:,3)+zvofs;
                      
                      v(:,1)=v(:,1)+tiley1-1;
                      v(:,2)=v(:,2)+tilex1-1;
                      if (vdata.data.exportobj.slicestep==1)
                        v(:,3)=v(:,3)+tilez1-1;
                      else
                        v(:,3)=((v(:,3)-0.5)*vdata.data.exportobj.slicestep)+0.5+firstblockslice-1;
                      end
                      v(:,1)=v(:,1)*param.yscale*param.yunit*mipfacty;
                      v(:,2)=v(:,2)*param.xscale*param.xunit*mipfactx;
                      v(:,3)=v(:,3)*param.zscale*param.zunit*mipfactz;
                    end
                    idx=tz*param.nrytiles*param.nrxtiles+ty*param.nrxtiles+tx;
                    param.fvindex(val(pcolnr),idx)=blocknr;
                    param.farray{blocknr}=f;
                    param.varray{blocknr}=v;
                    blocknr=blocknr+1;
                  end
                  
                  
                  colnr=endcolnr+1;
                end
                
                
              end
              
            else
              if ((param.extractwhich==6)||(param.extractwhich==7)||(param.extractwhich==8)||(param.extractwhich==9))
                cube=uint8((int32(rcube)+int32(gcube)+int32(bcube))/3);
              end
              obj=1;
              while ((obj<=size(param.objects,1))&&(vdata.state.lastcancel==0))
                if (param.extractwhich==5)
                  switch obj
                    case 1
                      subseg=double(rcube>128);
                    case 2
                      subseg=double(gcube>128);
                    case 3
                      subseg=double(bcube>128);
                  end
                end
                if ((param.extractwhich==6)||(param.extractwhich==7)||(param.extractwhich==8)||(param.extractwhich==9))
                  subseg=double(cube>param.lev(obj));
                end
                if (size(size(subseg),2)~=3)
                  f=[]; v=[];
                else
                  [f,v]=isosurface(subseg,0.5);
                end
                if (size(v,1)>0)
                  %adjust coordinates for bbox and when we added empty slices at beginning
                  v(:,1)=v(:,1)+yvofs;
                  v(:,2)=v(:,2)+xvofs;
                  v(:,3)=v(:,3)+zvofs;
                  
                  v(:,1)=v(:,1)+tiley1-1;
                  v(:,2)=v(:,2)+tilex1-1;
                  if (vdata.data.exportobj.slicestep==1)
                    v(:,3)=v(:,3)+tilez1-1;
                  else
                    v(:,3)=((v(:,3)-0.5)*vdata.data.exportobj.slicestep)+0.5+firstblockslice-1;
                  end
                  v(:,1)=v(:,1)*param.yscale*param.yunit*mipfacty;
                  v(:,2)=v(:,2)*param.xscale*param.xunit*mipfactx;
                  v(:,3)=v(:,3)*param.zscale*param.zunit*mipfactz;
                end
                param.farray{obj,tx,ty,tz}=f;
                param.varray{obj,tx,ty,tz}=v;
                obj=obj+1;
              end
            end
          end
        end
        
        tilex1=tilex1+param.blocksizex-param.overlap;
        tx=tx+1;
      end
      tiley1=tiley1+param.blocksizey-param.overlap;
      ty=ty+1;
    end
    tilez1=tilez1+param.blocksizez-param.overlap;
    tz=tz+1;
  end
  
  if (extractseg)
    vdata.vast.setsegtranslation([],[]);
  end
  
  if (vdata.state.lastcancel==0)
    message={'Exporting Surfaces ...', 'Merging meshes...'};
    set(vdata.ui.message,'String',message);
    pause(0.01);
    
    if (extractseg)
      param.objectsurfacearea=zeros(size(objects,1),1);
      switch vdata.data.exportobj.objectcolors
        case 1  %actual object colors
          colors=zeros(size(param.objects,1),3);
          for segnr=1:1:size(param.objects,1)
            seg=param.objects(segnr,1);
            %Get color from where the color is currently inherited from
            inheritseg=data(seg,18);
            colors(seg,:)=data(inheritseg, 3:5);
          end
        case 2  %colors from volume
          j=jet(256);
          vols=1+255*vdata.data.measurevol.lastvolume/max(vdata.data.measurevol.lastvolume);
          cols=j(round(vols),:);
          objs=vdata.data.measurevol.lastobjects(:,1);
          colors=zeros(size(param.objects,1),3); %vcols=zeros(nro,3);
          colors(objs,:)=cols*255;
      end
    else
      if (param.extractwhich==5)
        colors=zeros(size(param.objects,1),3);
        colors(1,1)=255;
        colors(2,2)=255;
        colors(3,3)=255;
      end
      if ((param.extractwhich==6)||(param.extractwhich==7)||(param.extractwhich==8)||(param.extractwhich==9))
        colors=[param.lev' param.lev' param.lev'];
      end
      if (param.extractwhich==10)
        %ids are colors in this case.
      end
    end


    %Write 3dsmax bulk loader script
    if (vdata.data.exportobj.write3dsmaxloader==1)
      save3dsmaxloader([param.targetfolder 'loadallobj_here.ms']);
    end
    
    %Merge full objects from components
    if (param.extractwhich==10)
      param.objects=find(param.objectvolume>0);
    end
    
    segnr=1;
    while ((segnr<=size(param.objects,1))&&(vdata.state.lastcancel==0))
      seg=param.objects(segnr,1);
      if (param.extractwhich==10)
        set(vdata.ui.message,'String',{'Exporting Surfaces ...', sprintf('Merging parts of object %d / %d...',segnr,length(param.objects) )});
        segname=sprintf('col_%02X%02X%02X',bitand(bitshift(seg,-16),255),bitand(bitshift(seg,-8),255),bitand(seg,255));
      else
        set(vdata.ui.message,'String',{'Exporting Surfaces ...', ['Merging parts of ' name{seg} '...']});
        segname=name{seg};
      end
      pause(0.01);
      
      cofp=[];
      covp=[];
      vofs=0;

      z=1;
      while ((z<=param.nrztiles)&&(vdata.state.lastcancel==0))
        y=1;
        while ((y<=param.nrytiles)&&(vdata.state.lastcancel==0))
          x=1;
          while ((x<=param.nrxtiles)&&(vdata.state.lastcancel==0))
            if (param.extractwhich==10)
              idx=z*param.nrytiles*param.nrxtiles+y*param.nrxtiles+x;
              blocknr=full(param.fvindex(seg,idx));
              if (blocknr==0)
                if (x==1)
                  f=[];
                  v=[];
                else
                end
              else
                
                if (x==1)
                  f=param.farray{blocknr}; %param.farray{seg,x,y,z};
                  v=param.varray{blocknr}; %param.varray{seg,x,y,z};
                else
                  [f,v]=mergemeshes(f,v,param.farray{blocknr},param.varray{blocknr});
                end
              end
            else
              if (x==1)
                f=param.farray{seg,x,y,z};
                v=param.varray{seg,x,y,z};
              else
                %disp(sprintf('Merging object %d, cube (%d,%d,%d)...',seg,x,y,z));
                [f,v]=mergemeshes(f,v,param.farray{seg,x,y,z},param.varray{seg,x,y,z});
              end
            end
            x=x+1;
          end
          if (y==1)
            fc=f;
            vc=v;
          else
            %disp(sprintf('Merging object %d, row (%d,%d)...',seg,y,z));
            [fc,vc]=mergemeshes(fc,vc,f,v);
          end
          y=y+1;
        end
        if (z==1)
          fp=fc;
          vp=vc;
        else
          %disp(sprintf('Merging object %d, plane %d...',seg,z));
          [fp,vp]=mergemeshes(fp,vp,fc,vc);
          
          %Take out non-overlapping part of matrices to speed up computation
          if ((size(vp,1)>1)&&(size(fp,1)>1))
            vcut=find(vp(:,3)==max(vp(:,3)),1,'first')-1;
            fcutind=find(fp>vcut,1,'first');
            [fcut,j]=ind2sub(size(fp),fcutind); fcut=fcut-1;
          
            covp=[covp; vp(1:vcut,:)]; vp=vp(vcut+1:end,:);
            ovofs=vofs;
            vofs=vofs+vcut;
            cofp=[cofp; fp(1:fcut,:)+ovofs]; fp=fp(fcut+1:end,:)-vcut;
          end
        end
        z=z+1;
      end
      
      vp=[covp; vp];
      fp=[cofp; fp+vofs];

      %invert Z axis if requested
      if (vdata.data.exportobj.invertz==1)
        if (size(vp,1)>0)
          vp(:,3)=-vp(:,3);
        end
      end
      
      %add offset if requested
      if (param.outputoffsetx~=0)
        vp(:,1)=vp(:,1)+param.outputoffsetx;
      end
      if (param.outputoffsety~=0)
        vp(:,2)=vp(:,2)+param.outputoffsety;
      end
      if (param.outputoffsetz~=0)
        vp(:,3)=vp(:,3)+param.outputoffsetz;
      end
      
      if (extractseg)
        on=name{find(data(:,1)==seg)};
      else
        on=segname;
      end
      on(on==' ')='_';
      on(on=='?')='_';
      on(on=='*')='_';
      on(on=='\\')='_';
      on(on=='/')='_';
      on(on=='|')='_';
      on(on==':')='_';
      on(on=='"')='_';
      on(on=='<')='_';
      on(on=='>')='_';
      

      if ((vdata.data.exportobj.skipmodelgeneration==0)&&(max(size(vp))>0))

        if (vdata.data.exportobj.fileformat==1) 
          %.OBJ
          filename=[param.targetfolder param.targetfileprefix sprintf('_%04d_%s.obj',seg,on)];

          objectname=[param.targetfileprefix sprintf('_%04d_%s',seg,segname)];
          
          mtlfilename=[param.targetfileprefix sprintf('_%04d_%s.mtl',seg,on)];
          mtlfilenamewithpath=[filename(1:end-3) 'mtl'];
          materialname=[param.targetfileprefix sprintf('_%04d_material',seg)];

          set(vdata.ui.message,'String',{'Exporting Surfaces ...', ['Saving ' filename ' as Wavefront OBJ.....']});
          pause(0.01);
          
          if (vdata.data.exportobj.invertz==1)
            vertface2obj_mtllink(vp,fp,filename,objectname,mtlfilename,materialname);
          else
            vertface2obj_mtllink_invnormal(vp,fp,filename,objectname,mtlfilename,materialname);
          end
          
          if (param.extractwhich==10)
            col=[bitand(bitshift(seg,-16),255) bitand(bitshift(seg,-8),255) bitand(seg,255)]/255;
            savematerialfile(mtlfilenamewithpath,materialname,col,1.0,0);
          else
            savematerialfile(mtlfilenamewithpath,materialname,colors(seg,:)/255,1.0,0);
          end
        end
        
        if (vdata.data.exportobj.fileformat==2) 
          %.PLY
          col=uint8(colors(seg,:));
          filename=[param.targetfolder param.targetfileprefix sprintf('_%04d_%s.ply',seg,on)];
          set(vdata.ui.message,'String',{'Exporting Surfaces ...', ['Saving ' filename ' as PLY file.....']});
          pause(0.01);
          savemeshtoply(vp,fp,filename,col(1),col(2),col(3));
        end
      end
      
      %param.vparray{seg}=vp;
      %param.fparray{seg}=fp;
      
      %%%%%% Compute surface size
      if (vdata.data.exportobj.savesurfacestats==1)
        %set(vdata.ui.message,'String',{'Exporting Surfaces ...', ['Evaluating surface area of ' name{seg} ' ...']});
        set(vdata.ui.message,'String',{'Exporting Surfaces ...', ['Evaluating surface area of ' segname ' ...']});
        pause(0.01);
        if (min(size(vp))>0)
          tnr=segnr;
          for tri=1:1:size(fp,1)
            v0=vp(fp(tri,1),:);
            v1=vp(fp(tri,2),:);
            v2=vp(fp(tri,3),:);
            a=cross(v1-v0,v2-v0); %abs not necessary because the values are squared later
            param.objectsurfacearea(tnr)=param.objectsurfacearea(tnr)+sqrt(sum(a.*a))/2;
          end
        end
      end
      
      segnr=segnr+1;
    end
  end
  
  if ((vdata.state.lastcancel==0)&&(vdata.data.exportobj.savesurfacestats==1))
    %write surface area values to text file
    fid = fopen([param.targetfolder vdata.data.exportobj.surfacestatsfile], 'wt');
    if (fid>0)
      fprintf(fid,'%% VastTools Surface Area Export\n');
      fprintf(fid,'%% Provided as-is, no guarantee for correctness!\n');
      fprintf(fid,'%% %s\n\n',get(vdata.fh,'name'));
      
      fprintf(fid,'%% Source File: %s\n',seglayername);
      fprintf(fid,'%% Mode: %s\n', vdata.data.exportobj.exportmodestring);
      fprintf(fid,'%% Area: (%d-%d, %d-%d, %d-%d)\n',rparam.xmin,rparam.xmax,rparam.ymin,rparam.ymax,rparam.zmin,rparam.zmax);
      fprintf(fid,'%% Computed at voxel size: (%f,%f,%f)\n',param.xscale*param.xunit*mipfactx,param.yscale*param.yunit*mipfacty,param.zscale*param.zunit*vdata.data.exportobj.slicestep*mipfactz);
      fprintf(fid,'%% Columns are: Object Name, Object ID, Surface Area in Export\n\n');
      for segnr=1:1:size(param.objects,1)
        seg=param.objects(segnr,1);
        if (param.extractwhich==10)
          segname=sprintf('col_%02X%02X%02X',bitand(bitshift(seg,-16),255),bitand(bitshift(seg,-8),255),bitand(seg,255));
          fprintf(fid,'"%s"  %d  %f\n',segname,seg,param.objectsurfacearea(segnr));
        else
          fprintf(fid,'"%s"  %d  %f\n',name{seg},seg,param.objectsurfacearea(segnr));
        end
      end
      fprintf(fid,'\n');
      fclose(fid);
    end
  end
  
  if (vdata.state.lastcancel==0)
    set(vdata.ui.message,'String','Done.');
  else
    set(vdata.ui.message,'String','Canceled.');
  end
  set(vdata.ui.cancelbutton,'Enable','off');
  vdata.state.lastcancel=0;
  

