import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'ROS 2',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Learn the Robot Operating System (ROS 2) framework for building robotic applications.
        Master communication protocols, package management, and development tools.
      </>
    ),
  },
  {
    title: 'Simulation',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Explore simulation environments like Gazebo and Unity for testing robotics algorithms
        in safe, controlled virtual worlds before deploying to real hardware.
      </>
    ),
  },
  {
    title: 'NVIDIA Isaac™',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Develop AI-powered robotics applications using NVIDIA Isaac™ platform.
        Leverage GPU acceleration for computer vision, perception, and control.
      </>
    ),
  },
  {
    title: 'VLA (Vision-Language-Action)',
    Svg: require('@site/static/img/undraw_vla.svg').default,
    description: (
      <>
        Build multimodal AI systems that integrate vision, language, and action.
        Create robots that understand natural language commands and perceive their environment.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className={styles.featureCard}>
        <div className="text--center">
          <Svg className={styles.featureSvg} role="img" />
        </div>
        <div className="text--center padding-horiz--md">
          <Heading as="h3" className={styles.featureTitle}>{title}</Heading>
          <p className={styles.featureDescription}>{description}</p>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
