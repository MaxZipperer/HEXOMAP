#include <stdio.h>
const float PI = 3.14159265359;
const float HALFPI = 0.5*PI;
/*
 * CHESS geometry: single detector, box beam, beam at the center of the detector, 360 degree rotation.
 * Same layout and argument order as device_code.cu; only the eta, chi and omega caps differ:
 *   chi:   full range [0, pi], G vectors pointing below the horizontal plane are kept.
 *   eta:   full range (-pi, pi], from cos(eta) = cos(chi)/cos(theta), limited by |eta| <= fEtaLimit.
 *   omega: limited by the detector omega window afDetInfo[17]..afDetInfo[18] instead of [-pi/2, pi/2].
 */

__device__ bool GetScatteringOmegas( float &fOmegaRes1, float &fOmegaRes2,
		float &fTwoTheta,float &fEta,float &fChi,
        const float *aScatteringVec,const float &fBeamEnergy){
	// identical to GetScatteringOmegas in device_code.cu
  float fScatteringVecMag = sqrt(aScatteringVec[0]*aScatteringVec[0] + aScatteringVec[1]*aScatteringVec[1] +aScatteringVec[2]*aScatteringVec[2]);

  float fSinTheta = fScatteringVecMag / ( (float)2.0 * 0.506773182 * fBeamEnergy);   // Bragg angle
  float fCosChi = aScatteringVec[2] / fScatteringVecMag;             // Tilt angle of G relative to z-axis
  float fSinChi = sqrt( 1.f - fCosChi * fCosChi );

  if( fabsf( fSinTheta ) <= fabsf( fSinChi) )
  {
	float fSinPhi = sin(acosf(fSinTheta / fSinChi));
	fEta = asinf(fSinChi * fSinPhi / sqrt( 1.f - fSinTheta * fSinTheta));
	// [-pi:pi]: angle to bring G to nominal position along +y-axis
	float fDeltaOmega0 = atan2f( aScatteringVec[0], aScatteringVec[1]);

	//  [0:pi/2] since arg >0: phi goes from above to Bragg angle
	float fDeltaOmega_b1 = asinf( fSinTheta/fSinChi );

	fOmegaRes1 = fDeltaOmega_b1 + fDeltaOmega0;  // oScatteringVec.m_fY > 0
	fOmegaRes2 = PI - fDeltaOmega_b1 + fDeltaOmega0;  // oScatteringVec.m_fY < 0
	if ( fOmegaRes1 > PI )          // range really doesn't matter
	  fOmegaRes1 -=  2.f * PI;

	if ( fOmegaRes1 < -PI)
	  fOmegaRes1 +=  2.f * PI;

	if ( fOmegaRes2 > PI)
	  fOmegaRes2 -= 2.f * PI;

	if ( fOmegaRes2 < -PI)
	  fOmegaRes2 += 2.f * PI;
	fTwoTheta = 2.f * asinf(fSinTheta);
	fChi = acosf(fCosChi);
	return true;
  }
  else
  {
	fOmegaRes1 = fOmegaRes2 = 0;     // too close to rotation axis to be illumination
	fTwoTheta = fEta = fChi = 0;
	return false;
  }
}

__device__ bool GetPeak_CHESS(int &iJ1,int &iJ2,int &iK1, int &iK2,bool &bHit1,bool &bHit2,
		const float &fOmegaRes1, const float &fOmegaRes2,
		const float &fTwoTheta, const float &fEta,const float &fChi,const float &fEtaLimit,
		const float *afVoxelPos,const float* __restrict__ afDetInfo){
	/*
	 * same as GetPeak in device_code.cu, except:
	 *   no chi cap, eta is signed over (-pi, pi], omega window comes from afDetInfo[17], afDetInfo[18].
	 * fEta (positive asin value from GetScatteringOmegas) is not used.
	 * afDetInfo:   	  	int iNPixelJ=0, iNPixelK=1;
    						float fPixelJ=2, fPixelK=3;
							float afCoordOrigin[3]=[4,5,6];
							float afNorm[3]=[7,8,9];
							float afJVector[3][10,11,12];
							float afKVector[3]=[13,14,15];
							float fNRot=16, fAngleStart=17, fAngleEnd=18;
	 */
	bHit1 = false;
	bHit2 = false;
	// cos(eta) = cos(chi)/cos(theta); chi > pi/2 gives cos(eta) < 0, i.e. below the beam.
	float fCosEta = cos(fChi) / cos(0.5f*fTwoTheta);
	fCosEta = fminf(1.f, fmaxf(-1.f, fCosEta));
	float fSinEta = sqrt(1.f - fCosEta * fCosEta);
	float fEta1 = atan2f(fSinEta, fCosEta);     // pairs with fOmegaRes1, equals +fEta when chi < pi/2
	float fEta2 = atan2f(-fSinEta, fCosEta);    // pairs with fOmegaRes2, equals -fEta when chi < pi/2
	float fVoxelPosX,fVoxelPosY,fVoxelPosZ,fDist,fAngleNormScatter;
	int iJ,iK;
	float afScatterDir[3]; //scattering direction
	float afInterPos[3];
	if ((afDetInfo[17]<=fOmegaRes1) && (fOmegaRes1<=afDetInfo[18]) && (fabsf(fEta1)<=fEtaLimit)){
		fVoxelPosX = cos(fOmegaRes1)*afVoxelPos[0] - sin(fOmegaRes1)*afVoxelPos[1];
		fVoxelPosY = cos(fOmegaRes1)*afVoxelPos[1] + sin(fOmegaRes1)*afVoxelPos[0];
		fVoxelPosZ = afVoxelPos[2];
		fDist = afDetInfo[7]*(afDetInfo[4] - fVoxelPosX)
				+ afDetInfo[8]*(afDetInfo[5] - fVoxelPosY)
				+ afDetInfo[9]*(afDetInfo[6] - fVoxelPosZ);
		afScatterDir[0] = cos(fTwoTheta);
		afScatterDir[1] = sin(fTwoTheta) * sin(fEta1);
		afScatterDir[2] = sin(fTwoTheta) * cos(fEta1);
		fAngleNormScatter = afDetInfo[7]*afScatterDir[0]
		                          + afDetInfo[8]*afScatterDir[1]
		                          + afDetInfo[9]*afScatterDir[2];
		afInterPos[0] = fDist / fAngleNormScatter * afScatterDir[0] + fVoxelPosX;
		afInterPos[1] = fDist / fAngleNormScatter * afScatterDir[1] + fVoxelPosY;
		afInterPos[2] = fDist / fAngleNormScatter * afScatterDir[2] + fVoxelPosZ;
		iJ = floor((afDetInfo[10]*(afInterPos[0]-afDetInfo[4])
				+ afDetInfo[11]*(afInterPos[1]-afDetInfo[5])
				+ afDetInfo[12]*(afInterPos[2]-afDetInfo[6]) )/afDetInfo[2]);
		iK = floor((afDetInfo[13]*(afInterPos[0]-afDetInfo[4])
				+ afDetInfo[14]*(afInterPos[1]-afDetInfo[5])
				+ afDetInfo[15]*(afInterPos[2]-afDetInfo[6]) )/afDetInfo[3]);
		bHit1 = (0<=iJ ) && (iJ<afDetInfo[0]) && (0<=iK) && (iK<afDetInfo[1]);
		iJ1 = iJ;
		iK1 = iK;
	}

	if ((afDetInfo[17]<=fOmegaRes2) && (fOmegaRes2<=afDetInfo[18]) && (fabsf(fEta2)<=fEtaLimit)){
		fVoxelPosX = cos(fOmegaRes2)*afVoxelPos[0] - sin(fOmegaRes2)*afVoxelPos[1];
		fVoxelPosY = cos(fOmegaRes2)*afVoxelPos[1] + sin(fOmegaRes2)*afVoxelPos[0];
		fVoxelPosZ = afVoxelPos[2];
		fDist = afDetInfo[7]*(afDetInfo[4] - fVoxelPosX)
				+ afDetInfo[8]*(afDetInfo[5] - fVoxelPosY)
				+ afDetInfo[9]*(afDetInfo[6] - fVoxelPosZ);
		afScatterDir[0] = cos(fTwoTheta);
		afScatterDir[1] = sin(fTwoTheta) * sin(fEta2);
		afScatterDir[2] = sin(fTwoTheta) * cos(fEta2);
		fAngleNormScatter = afDetInfo[7]*afScatterDir[0]
		                          + afDetInfo[8]*afScatterDir[1]
		                          + afDetInfo[9]*afScatterDir[2];
		afInterPos[0] = fDist / fAngleNormScatter * afScatterDir[0] + fVoxelPosX;
		afInterPos[1] = fDist / fAngleNormScatter * afScatterDir[1] + fVoxelPosY;
		afInterPos[2] = fDist / fAngleNormScatter * afScatterDir[2] + fVoxelPosZ;
		iJ = floor((afDetInfo[10]*(afInterPos[0]-afDetInfo[4])
				+ afDetInfo[11]*(afInterPos[1]-afDetInfo[5])
				+ afDetInfo[12]*(afInterPos[2]-afDetInfo[6]) )/afDetInfo[2]);
		iK = floor((afDetInfo[13]*(afInterPos[0]-afDetInfo[4])
				+ afDetInfo[14]*(afInterPos[1]-afDetInfo[5])
				+ afDetInfo[15]*(afInterPos[2]-afDetInfo[6]) )/afDetInfo[3]);
		bHit2 = (0<=iJ ) && (iJ<afDetInfo[0]) && (0<=iK) && (iK<afDetInfo[1]);
		iJ2 = iJ;
		iK2 = iK;
	}
	return true;

}

__global__ void simulation_CHESS(int *aiJ, int *aiK, float *afOmega, bool *abHit,int *aiRotN,
		const int iNVoxel, const int iNOrientation, const int iNG, const int iNDet,
		const float* afOrientation,const float* afVoxelPos,
		const float fBeamEnergy, const float fEtaLimit, const float* __restrict__ afDetInfo,
		const float* __restrict__ afG){
	/*
	 * same arguments, output layout and launch configuration as simulation in device_code.cu,
	 * so the output can be passed directly to hitratio_multi_detector.
	 * <<< (iNVoxel,iNOrientation),(iNG)>>>;
	 */
	float fOmegaRes1,fOmegaRes2,fTwoTheta,fEta,fChi;
	float afScatteringVec[3]={0,0,0};
	int iBase = blockIdx.x*gridDim.y*blockDim.x*2*iNDet+ blockIdx.y*blockDim.x*2*iNDet + threadIdx.x*2*iNDet;
	for(int d=0; d<2*iNDet; d++){
		abHit[iBase+d] = false;
		aiJ[iBase+d] = 0;
		aiK[iBase+d] = 0;
		aiRotN[iBase+d] = 0;
		afOmega[iBase+d] = 0.f;
	}
	for (int i=0;i<3;i++){
		for(int j=0;j<3;j++){
		    afScatteringVec[i] += afOrientation[blockIdx.x*gridDim.y*9+blockIdx.y*9+i*3+j]*afG[threadIdx.x*3+j];
		}
	}
	if(GetScatteringOmegas( fOmegaRes1, fOmegaRes2, fTwoTheta, fEta, fChi , afScatteringVec,fBeamEnergy)){
		// GetScatteringOmegas returns omega in (-pi, pi]; shift into [detOmegaStart, detOmegaStart + 2pi)
		if (fOmegaRes1 < afDetInfo[17])
			fOmegaRes1 += 2.f * PI;
		if (fOmegaRes2 < afDetInfo[17])
			fOmegaRes2 += 2.f * PI;
		int i = iBase;
		for(int iDetIdx=0;iDetIdx<iNDet;iDetIdx++){
			GetPeak_CHESS(aiJ[i+iDetIdx],aiJ[i+iDetIdx+iNDet],aiK[i+iDetIdx],aiK[i+iDetIdx+iNDet],
					abHit[i+iDetIdx],abHit[i+iDetIdx+iNDet],
					fOmegaRes1, fOmegaRes2,fTwoTheta,
					fEta,fChi,fEtaLimit,afVoxelPos+blockIdx.x*3,afDetInfo+19*iDetIdx);
			//	////////assuming they are using the same rotation number in all the detectors!!!!!!!///////////////////
			afOmega[i+iDetIdx] = fOmegaRes1;
			aiRotN[i+iDetIdx] = floor((fOmegaRes1-afDetInfo[17])/(afDetInfo[18]-afDetInfo[17])*(afDetInfo[16]-1));
			afOmega[i+iDetIdx+iNDet] = fOmegaRes2;
			aiRotN[i+iDetIdx+iNDet] = floor((fOmegaRes2-afDetInfo[17])/(afDetInfo[18]-afDetInfo[17])*(afDetInfo[16]-1));
		}
	}
}
