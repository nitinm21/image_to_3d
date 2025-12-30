declare module "@mkkellogg/gaussian-splats-3d" {
  import * as THREE from "three";

  export enum SceneRevealMode {
    Default = 0,
    Gradual = 1,
    Instant = 2,
  }

  export interface ViewerOptions {
    renderer?: THREE.WebGLRenderer;
    camera?: THREE.PerspectiveCamera;
    selfDrivenMode?: boolean;
    useBuiltInControls?: boolean;
    sharedMemoryForWorkers?: boolean;
    dynamicScene?: boolean;
    sceneRevealMode?: SceneRevealMode;
    antialiased?: boolean;
    focalAdjustment?: number;
  }

  export interface SplatSceneOptions {
    splatAlphaRemovalThreshold?: number;
    showLoadingUI?: boolean;
    progressiveLoad?: boolean;
    onProgress?: (progress: number) => void;
  }

  export class Viewer {
    constructor(options?: ViewerOptions);
    addSplatScene(
      url: string,
      options?: SplatSceneOptions
    ): Promise<void>;
    update(): void;
    render(): void;
    dispose(): void;
  }
}
